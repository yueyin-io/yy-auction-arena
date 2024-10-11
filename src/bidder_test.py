import unittest
from .bidder_base import Bidder  # Adjust the import based on your file structure
from .item_base import Item  # Assuming you have an Item class defined
import openai # Import the OpenAI API client
import os

class TestBidderFactory(unittest.TestCase):

    def test_create_bidder(self):
        """
        Test the create factory method for the Bidder class.
        """
        data = {
            "name": "TestBidder",
            "model_name": "gpt-3.5-turbo",
            "budget": 1000,
            "desire": "maximize_profit",
            "plan_strategy": "static",
            "temperature": 0.5,
            "correct_belief": True
        }
        bidder = Bidder.create(**data)
        
        # Check if the instance was created with the correct attributes
        self.assertEqual(bidder.name, "TestBidder")
        self.assertEqual(bidder.budget, 1000)
        self.assertEqual(bidder.model_name, "gpt-3.5-turbo")
        self.assertEqual(bidder.system_message, f"Your primary objective is to secure the highest profit at the end of this auction, compared to all other bidders")

        # Check if the post-init values are set correctly
        self.assertEqual(bidder.original_budget, 1000)
        self.assertTrue(len(bidder.dialogue_history) > 0)

class TestBidderHelperMethods(unittest.TestCase):

    def setUp(self):
        """Create a test bidder and a set of items for testing."""
        self.items = [
            Item(name="Item 1", price=100, true_value=150),
            Item(name="Item 2", price=200, true_value=300),
            Item(name="Item 3", price=150, true_value=200),
        ]
        self.bidder = Bidder.create(
            name="TestBidder",
            model_name="gpt-3.5-turbo",
            budget=1000,
            desire="maximize_profit",
            plan_strategy="static",
            temperature=0.7,
            correct_belief=True
        )
        self.bidder.items = self.items

    def test_get_estimated_value(self):
        """Test if the estimated value calculation is correct."""
        estimated_value = self.bidder._get_estimated_value(self.items[0])
        self.assertEqual(estimated_value, 165)  # 150 * 1.1 = 165

    def test_get_cur_item(self):
        """Test retrieving the current item."""
        self.bidder.cur_item_id = 0
        cur_item = self.bidder._get_cur_item()
        self.assertEqual(cur_item.name, "Item 1")

        # Test with a specific key
        cur_item_price = self.bidder._get_cur_item("price")
        self.assertEqual(cur_item_price, 100)

    def test_get_next_item(self):
        """Test retrieving the next item."""
        self.bidder.cur_item_id = 0
        next_item = self.bidder._get_next_item()
        self.assertEqual(next_item.name, "Item 2")

        # Test when there is no next item
        self.bidder.cur_item_id = 2
        next_item = self.bidder._get_next_item()
        self.assertEqual(next_item, "no item left")

    def test_get_remaining_items(self):
        """Test retrieving all remaining items."""
        self.bidder.cur_item_id = 1
        remaining_items = self.bidder._get_remaining_items()
        self.assertEqual(len(remaining_items), 1)  # Should only contain "Item 3"

        # Test with string output
        remaining_items_str = self.bidder._get_remaining_items(as_str=True)
        self.assertEqual(remaining_items_str, "Item 3")

    def test_get_items_value_str(self):
        """Test the formatted string for item values."""
        value_str = self.bidder._get_items_value_str(self.items)
        expected_str = (
            "1. Item 1, starting price is $100. Your estimated value for this item is $165.\n"
            "2. Item 2, starting price is $200. Your estimated value for this item is $330.\n"
            "3. Item 3, starting price is $150. Your estimated value for this item is $220."
        )
        self.assertEqual(value_str, expected_str)

class TestBidderRealCalls(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up OpenAI API key and configurations."""
        # Set up the OpenAI API key from the environment
        cls.api_key = os.getenv('OPENAI_API_KEY')
        if not cls.api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = cls.api_key

    def setUp(self):
        """Set up the Bidder instance and items for testing."""
        self.items = [
            Item(name="Item 1", price=100, true_value=150),
            Item(name="Item 2", price=200, true_value=300),
            Item(name="Item 3", price=150, true_value=200),
        ]
        # Create a Bidder instance
        self.bidder = Bidder.create(
            name="TestBidder",
            model_name="gpt-4",  # Change to the appropriate model
            budget=500,
            desire="maximize_profit",
            plan_strategy="dynamic",
            temperature=0.7,
            correct_belief=True
        )
        self.bidder.items = self.items

    def test_learn_from_prev_auction_real_llm(self):
        """Test the learning process from previous auctions using a real LLM call."""
        # Define past learnings and auction log
        past_learnings = "Overbidding led to budget constraints."
        past_auction_log = "Auction Log: Competitor A bid very aggressively on Item 2."

        # Call the method with real LLM interaction
        result = self.bidder.learn_from_prev_auction(past_learnings, past_auction_log)

        # Print the LLM output for validation
        print("LLM Learnings Output: ", result)

        # Loose assertion to check key expected phrases in the output
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("overbidding", result.lower())

    def test_choose_items_real_llm(self):
        """Test item selection for 'maximize_profit' strategy."""
        # Use the _choose_items method with a real scenario
        selected_items = self.bidder._choose_items(budget=500, items=self.items)

        # Print selected items
        print("Selected Items for Maximize Profit: ", [item.name for item in selected_items])

        # Assertions based on expected behavior
        self.assertEqual(len(selected_items), 2)  # Should select 2 items within budget
        self.assertEqual(selected_items[0].name, "Item 2")  # Highest value item first

    def test_get_plan_instruct_real_llm(self):
        """Test generating a plan instruction string for bidding."""
        # Generate the plan instruction
        plan_instruct = self.bidder.get_plan_instruct(self.items)

        # Print the plan instruction
        print("Generated Plan Instruction: ", plan_instruct)

        # Basic assertions to verify output
        self.assertIn("TestBidder", plan_instruct)
        self.assertIn("Your estimated value for this item is", plan_instruct)

    def test_init_plan_real_llm(self):
        """Test initialization of a plan using real LLM interaction."""
        # Generate the plan instruction
        plan_instruct = self.bidder.get_plan_instruct(self.items)

        # Call the init_plan method to get real LLM output
        result = self.bidder.init_plan(plan_instruct)

        # Print the LLM's generated plan
        print("LLM Generated Plan: ", result)

        # Assertions to verify the plan is non-empty and correctly formatted
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("Item 1", result)
        self.assertIn("Item 2", result)

class TestBidderBiddingMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up OpenAI API key and configurations."""
        # Set up the OpenAI API key from the environment or specify directly
        cls.api_key = os.getenv('OPENAI_API_KEY')
        if not cls.api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = cls.api_key

    def setUp(self):
        """Set up the Bidder instance and items for testing."""
        self.items = [
            Item(name="Item 1", price=100, true_value=150),
            Item(name="Item 2", price=200, true_value=300),
            Item(name="Item 3", price=150, true_value=200),
        ]
        # Create a Bidder instance
        self.bidder = Bidder.create(
            name="TestBidder",
            model_name="gpt-4",  # Specify the LLM model to use
            budget=500,
            desire="maximize_profit",
            plan_strategy="dynamic",
            temperature=0.7,
            correct_belief=True
        )
        self.bidder.items = self.items

    def test_get_rebid_instruct(self):
        """Test constructing a rebid instruction."""
        auctioneer_msg = "The current bid is $200. Do you want to increase your bid?"
        result = self.bidder.get_rebid_instruct(auctioneer_msg)

        # Print and validate the result
        print("Rebid Instruction: ", result)
        self.assertEqual(result, auctioneer_msg)
        self.assertEqual(self.bidder.dialogue_history[-2]['content'], auctioneer_msg)

    def test_get_bid_instruct(self):
        """Test constructing a bid instruction."""
        auctioneer_msg = "The current highest bid is $150 for Item 2."
        bid_round = 1

        # Call the method and print the generated instruction
        bid_instruction = self.bidder.get_bid_instruct(auctioneer_msg, bid_round)
        print("Generated Bid Instruction: ", bid_instruction)

        # Basic assertions to check if the instruction is formatted correctly
        self.assertIn("The current highest bid is $150 for Item 2.", bid_instruction)
        self.assertIn("Your primary objective is to secure the highest profit", bid_instruction)

    def test_bid_rule(self):
        """Test rule-based bidding strategy."""
        # Simulate a scenario with the current highest bid at $120
        cur_bid = 120
        next_bid = self.bidder.bid_rule(cur_bid, min_markup_pct=0.1)

        # Print and validate the bidding output
        print(f"Rule-based bid: ${next_bid}")
        self.assertEqual(next_bid, 135)  # 120 + 10% markup of 150 (next bid should be 135)
        self.assertIn("I bid $135!", self.bidder.dialogue_history[-1]['content'])

    def test_bid_real_llm(self):
        """Test full bidding process with real LLM interaction."""
        # Construct the bidding instruction
        auctioneer_msg = "The current highest bid is $150 for Item 1."
        bid_instruction = self.bidder.get_bid_instruct(auctioneer_msg, bid_round=0)

        # Call the bid method with real LLM interaction
        llm_response = self.bidder.bid(bid_instruction)

        # Print the LLM's generated bid response
        print("LLM Bid Response: ", llm_response)

        # Assert that the LLM response is non-empty and contains relevant bid information
        self.assertIsInstance(llm_response, str)
        self.assertGreater(len(llm_response), 0)
        self.assertIn("Item", llm_response)

class TestBidderSummarizeAndReplan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up OpenAI API key and configurations."""
        # Set up the OpenAI API key from the environment or specify directly
        cls.api_key = os.getenv('OPENAI_API_KEY')
        if not cls.api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = cls.api_key

    def setUp(self):
        """Set up the Bidder instance and items for testing."""
        self.items = [
            Item(name="Item 1", price=100, true_value=150),
            Item(name="Item 2", price=200, true_value=300),
            Item(name="Item 3", price=150, true_value=200),
        ]
        # Create a Bidder instance
        self.bidder = Bidder.create(
            name="TestBidder",
            model_name="gpt-4",  # Specify the LLM model to use
            budget=500,
            desire="maximize_profit",
            plan_strategy="dynamic",
            temperature=0.7,
            correct_belief=True
        )
        self.bidder.items = self.items

    def test_get_summarize_instruct(self):
        """Test the construction of summarize instruction strings."""
        bidding_history = "Bidder A bid $150, Bidder B bid $200, and Bidder C won with $250."
        hammer_msg = "Sold to Bidder C for $250."
        win_lose_msg = "You lost the bid."

        # Call the method
        instruction = self.bidder.get_summarize_instruct(bidding_history, hammer_msg, win_lose_msg)

        # Print and validate the instruction string
        print("Generated Summarize Instruction: ", instruction)
        self.assertIn("Bidder A bid $150", instruction)
        self.assertIn("You lost the bid", instruction)
        self.assertIn("Sold to Bidder C for $250", instruction)

    def test_summarize_real_llm(self):
        """Test the LLM interaction for summarizing auction results."""
        bidding_history = "Bidder A bid $150, Bidder B bid $200, and Bidder C won with $250."
        hammer_msg = "Sold to Bidder C for $250."
        win_lose_msg = "You lost the bid."

        # Create the summarize instruction using the real method.
        instruction = self.bidder.get_summarize_instruct(bidding_history, hammer_msg, win_lose_msg)

        # Call the summarize method, which uses real LLM interaction.
        llm_response = self.bidder.summarize(instruction)

        # Print and validate the LLM response
        print("LLM Summarize Response: ", llm_response)
        self.assertIsInstance(llm_response, str)
        self.assertGreater(len(llm_response), 0)
        self.assertIn("Bidder", llm_response)  # Check that the response is relevant

    def test_get_replan_instruct(self):
        """Test the construction of replan instructions."""
        # Generate a replan instruction based on current status quo
        instruction = self.bidder.get_replan_instruct()

        # Print and validate the replan instruction
        print("Generated Replan Instruction: ", instruction)
        self.assertIn("Current Status:", instruction)
        self.assertIn("Remaining Items:", instruction)
        self.assertIn("maximize_profit", instruction)  # Check that the correct desire is included

    def test_replan_real_llm(self):
        """Test the full replanning process using real LLM interaction."""
        # Create a replan instruction using the real method.
        replan_instruction = self.bidder.get_replan_instruct()

        # Call the replan method, which interacts with the real LLM.
        replan_response = self.bidder.replan(replan_instruction)

        # Print and validate the response
        print("LLM Replan Response: ", replan_response)
        self.assertIsInstance(replan_response, str)
        self.assertGreater(len(replan_response), 0)
        self.assertIn("priority", replan_response.lower())  # Check for relevant keywords in the response


if __name__ == '__main__':
    unittest.main()



