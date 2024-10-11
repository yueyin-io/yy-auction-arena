from typing import List
from langchain.base_language import BaseLanguageModel
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import (
    ChatAnthropic,
    ChatOpenAI,
    ChatVertexAI,
    ChatGooglePalm,
)
import vertexai
from langchain.input import get_colored_text
from langchain.callbacks import get_openai_callback
from collections import defaultdict
from pydantic import BaseModel
import queue
import threading
import os
import random
import time
import ujson as json
import matplotlib.pyplot as plt
from .item_base import Item, item_list_equal
from .prompt_base import (
    AUCTION_HISTORY,
    _LEARNING_STATEMENT,
    INSTRUCT_PLAN_TEMPLATE,
    INSTRUCT_BID_TEMPLATE,
    INSTRUCT_SUMMARIZE_TEMPLATE,
    INSTRUCT_LEARNING_TEMPLATE,
    INSTRUCT_REPLAN_TEMPLATE,
    SYSTEM_MESSAGE,
)
import sys
sys.path.append('..')
import openai
from utils import LoadJsonL, extract_jsons_from_text, extract_numbered_list, trace_back, get_num_tokens_from_messages

# Initialize the OpenAI client (specific to your API setup)
cilent = openai.OpenAI()

# DESIRE_DESC dictionary defines different bidding goals for agents. This is a simplified version of previous configurations.
DESIRE_DESC = {
    'maximize_profit': "Your primary objective is to secure the highest profit at the end of this auction, compared to all other bidders",
    'maximize_items': "Your primary objective is to win the highest number of items at the end of this auction, compared to everyone else",
}

# Define a Bidder class that models the behavior of a bidding agent in an auction.
# The Bidder class is a Pydantic BaseModel that defines the attributes and methods for a bidding agent.
class Bidder(BaseModel):
    name: str  # Bidder's name
    model_name: str  # Name of the model used for bidding (e.g., gpt-4, gpt-3.5)
    budget: int  # Total budget available for the bidder
    desire: str  # Bidding goal defined in DESIRE_DESC (maximize_profit, maximize_items)
    plan_strategy: str  # Strategy to plan bids (e.g., static, dynamic)
    temperature: float = 0.7  # Temperature parameter for controlling LLM responses
    overestimate_percent: int = 10  # Percentage for overestimating item value (adds a buffer)
    correct_belief: bool  # Boolean to determine whether to update beliefs based on new information
    enable_learning: bool = False  # Flag to enable or disable learning from previous bids

    # LLM-specific attributes
    llm: BaseLanguageModel = None  # Placeholder for LLM instance
    openai_cost = 0  # Accumulated cost for LLM usage
    llm_token_count = 0  # Track the number of tokens used in LLM calls

    # General auction attributes
    verbose: bool = False  # Flag to enable verbose logging
    auction_hash: str = ''  # Unique identifier for the auction

    system_message: str = ''  # Initial system message for the LLM
    original_budget: int = 0  # Store original budget for reference

    # Memory and tracking variables
    profit: int = 0  # Profit accumulated so far
    cur_item_id = 0  # Current item index in the auction
    items: list = []  # List of items available for bidding
    dialogue_history: list = []  # Conversation history between agent and LLM
    llm_prompt_history: list = []  # History of LLM prompts and responses
    items_won = []  # List of items won by this bidder
    bid_history: list = []  # History of all bids made in the current auction
    plan_instruct: str = ''  # Instruction for creating a bidding plan
    cur_plan: str = ''  # Current plan in text format
    status_quo: dict = {}  # Status of the auction, including beliefs about competitors
    withdraw: bool = False  # Flag indicating if the agent has withdrawn from bidding
    learnings: str = ''  # Learning points from previous auctions
    max_bid_cnt: int = 4  # Maximum number of bids allowed per item
    rule_bid_cnt: int = 0  # Counter for the number of bids made on the current item

    # Belief tracking for debugging and understanding agent behavior
    failed_bid_cnt: int = 0  # Count of failed bids (bids exceeding budget)
    total_bid_cnt: int = 0  # Total number of bids made
    self_belief_error_cnt: int = 0  # Errors in self-belief tracking
    total_self_belief_cnt: int = 0  # Total self-belief checks
    other_belief_error_cnt: int = 0  # Errors in beliefs about other agents
    total_other_belief_cnt: int = 0  # Total belief checks about other agents

    # Historical data tracking
    engagement_count: int = 0  # Count of active engagements in auctions
    budget_history = []  # History of budget changes
    profit_history = []  # History of profit changes
    budget_error_history = []  # History of budget estimation errors
    profit_error_history = []  # History of profit estimation errors
    win_bid_error_history = []  # History of winning bid errors
    engagement_history = defaultdict(int)  # Engagement history by item
    all_bidders_status = {}  # Track other bidders' profit and status
    changes_of_plan = []  # Log changes in strategy or plan

    # Additional placeholders for unused features
    input_box: str = None
    need_input = False
    semaphore = 0

    class Config:
        """Allow for arbitrary types in Pydantic BaseModel."""
        arbitrary_types_allowed = True

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
    
    @classmethod
    def create(cls, **data):
        """
        Factory method to create a new Bidder instance and initialize it.
        """
        instance = cls(**data)
        instance._post_init()
        return instance

    def _post_init(self):
        """
        Perform additional initialization after the instance is created.
        This method sets up the initial budget, system message, and LLM.
        """
        self.original_budget = self.budget
        # Format and set the system message with the desired goal
        self.system_message = SYSTEM_MESSAGE.format(
            name=self.name,
            desire_desc=DESIRE_DESC[self.desire],
        )
        self._parse_llm()
        # Initialize dialogue history with the system message
        self.dialogue_history += [
            {"role": "system", "content": self.system_message},
            {"role": "assistant", "content": ''}
        ]
        # Track initial budget and profit history
        self.budget_history.append(self.budget)
        self.profit_history.append(self.profit)

    def _parse_llm(self):
        """
        Set up the LLM client based on the model name.
        Currently supports only OpenAI models.
        """
        if 'gpt-' in self.model_name:
            self.llm = cilent.chat.completions.create(model=self.model_name,
                temperature=self.temperature,
                timeout=1200)
        else:
            raise NotImplementedError(f"Model {self.model_name} is not supported.")

    def _run_llm_standalone(self, input_messages: list):
        """
        Run the LLM in a standalone mode using the provided messages.
        This method handles rate limits and various exceptions.
        """
        for i in range(6):  # Retry up to 6 times if rate limit is exceeded
            try:
                input_token_num = get_num_tokens_from_messages(input_messages, self.model_name)
                if 'gpt-3.5-turbo' in self.model_name and '16k' not in self.model_name:
                    max_tokens = max(3900 - input_token_num, 192)
                elif 'gpt-4' in self.model_name:
                    max_tokens = max(8000 - input_token_num, 192)
                else:
                    raise NotImplementedError(f"Model {self.model_name} is not supported.")
                
                # Call the LLM and capture the response
                response = self.llm(messages=input_messages, max_tokens=max_tokens)
                result = response.choices[0].message['content'].strip()
                self.openai_cost += response['usage']['total_tokens']
                break
            except openai.error.RateLimitError:
                # Exponential backoff for rate limit errors
                wait_time = 2 ** (i + 1)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except openai.error.AuthenticationError:
                print("Authentication failed. Please check your API key.")
                return ""
            except openai.error.OpenAIError as e:
                print(f"An error occurred: {e}")
                return ""
        # Track the number of tokens used
        self.llm_token_count = get_num_tokens_from_messages(input_messages, self.model_name)
        return result


    def _get_estimated_value(self, item):
        """
        Calculate an overestimated value for an item based on the bidder's internal logic.

        Args:
            item (Item): The item object for which to calculate the estimated value.

        Returns:
            int: The estimated value for the item, overestimated by the defined percentage.

        Notes:
            This function multiplies the true value of the item by (1 + overestimate_percent / 100).
            The `overestimate_percent` is a class attribute that introduces a bias in estimating item value.
        """
        value = item.true_value * (1 + self.overestimate_percent / 100)
        return int(value)


    def _get_cur_item(self, key=None):
        """
        Retrieve the current item in the list of items for the bidder.

        Args:
            key (str, optional): If provided, retrieves a specific attribute of the current item.
        
        Returns:
            Any: The current item or a specific attribute if `key` is provided.
            str: Returns 'no item left' if the current item index exceeds the number of items.

        Usage:
            If `key` is not provided, it returns the item object at `cur_item_id`.
            If `key` is provided (e.g., "price"), it returns the attribute value for that key.
        """
        if self.cur_item_id < len(self.items):
            if key is not None:
                return self.items[self.cur_item_id].__dict__[key]
            else:
                return self.items[self.cur_item_id]
        else:
            return 'no item left'


    def _get_next_item(self, key=None):
        """
        Retrieve the next item in the list, relative to the current item.

        Args:
            key (str, optional): If provided, retrieves a specific attribute of the next item.
        
        Returns:
            Any: The next item object or a specific attribute if `key` is provided.
            str: Returns 'no item left' if there is no subsequent item in the list.

        Usage:
            This function is similar to `_get_cur_item` but looks one index ahead.
        """
        if self.cur_item_id + 1 < len(self.items):
            if key is not None:
                return self.items[self.cur_item_id + 1].__dict__[key]
            else:
                return self.items[self.cur_item_id + 1]
        else:
            return 'no item left'


    def _get_remaining_items(self, as_str=False):
        """
        Retrieve a list of all items remaining after the current item.

        Args:
            as_str (bool, optional): If True, returns a comma-separated string of remaining item names.
        
        Returns:
            list or str: Returns a list of remaining items or a string of item names if `as_str` is True.
        """
        remain_items = self.items[self.cur_item_id + 1:]
        if as_str:
            return ', '.join([item.name for item in remain_items])  # Create a string of item names
        else:
            return remain_items  # Return the remaining items as a list


    def _get_items_value_str(self, items: List[Item]):
        """
        Generate a formatted string describing the estimated values for a given list of items.

        Args:
            items (List[Item]): List of item objects to describe.

        Returns:
            str: A formatted string showing each item with its starting price and estimated value.

        Usage:
            This method is useful for generating textual instructions for planning or summary reports.
            The format is as follows:
                1. ItemName, starting price is $X. Your estimated value for this item is $Y.
                2. ...
        """
        if not isinstance(items, list):
            items = [items]  # Ensure items is a list

        items_info = ''
        for i, item in enumerate(items):
            estimated_value = self._get_estimated_value(item)  # Get the estimated value of the item
            _info = f"{i+1}. {item}, starting price is ${item.price}. Your estimated value for this item is ${estimated_value}.\n"
            items_info += _info
        return items_info.strip()

    
    # ********** Main Instructions and Functions ********** #
    
    def learn_from_prev_auction(self, past_learnings, past_auction_log):
        """
        Learn from previous auction experiences using an LLM.

        Args:
            past_learnings (str): Text representing the learnings from the previous auctions.
            past_auction_log (str): Text-based log of the auction's events.

        Returns:
            str: The summarized learnings generated by the LLM.

        Description:
            - This method leverages a pre-defined learning template (`INSTRUCT_LEARNING_TEMPLATE`) 
            to format the inputs and sends it to the LLM using `_run_llm_standalone`.
            - The response is then processed to extract key learning points and added to the dialogue history.
            - Learning points are stored in `self.learnings` and appended to the `system_message` for further guidance.

        Conditions:
            - Learning is disabled if `enable_learning` is False or if the model is labeled as 'rule' or 'human'.
        """
        if not self.enable_learning or 'rule' in self.model_name or 'human' in self.model_name:
            return ''

        # Prepare the instruction message for learning from past data.
        instruct_learn = INSTRUCT_LEARNING_TEMPLATE.format(
            past_auction_log=past_auction_log,
            past_learnings=past_learnings
        )

        # Create the LLM input message.
        user_input_messages = [{"role": "user", "content": instruct_learn}]
        
        # Run the LLM with the prepared message.
        result = self._run_llm_standalone(user_input_messages)
        
        # Track the dialogue history and LLM prompt history.
        self.dialogue_history += [
            user_input_messages,
            {"role": "assistant", "content": result}
        ]
        self.llm_prompt_history.append({
            'messages': user_input_messages,
            'result': result,
            'tag': 'learn_0'
        })
        
        # Extract key learning points and update the system message.
        self.learnings = '\n'.join(extract_numbered_list(result))
        if self.learnings != '':
            self.system_message += f"\n\nHere are your key learning points and practical tips from a previous auction. You can use them to guide this auction:\n```\n{self.learnings}\n```"

        # If verbose mode is enabled, print a log message.
        if self.verbose:
            print(f"Learn from previous auction: {self.name} ({self.model_name}).")
        return result


    def _choose_items(self, budget, items: List[Item]):
        """
        Choose items to bid on based on the bidder's budget and goal.

        Args:
            budget (int): The available budget for bidding.
            items (List[Item]): List of items to choose from.

        Returns:
            List[Item]: A list of items selected based on the budget and strategy.

        Description:
            - If `desire` is 'maximize_items', sorts items in ascending order by estimated value (cheapest first).
            - If `desire` is 'maximize_profit', sorts items in descending order by estimated value (most valuable first).
            - Iterates through the sorted list and selects items as long as the budget allows.
        """
        # Sort items based on estimated value.
        sorted_items = sorted(items, key=lambda x: self._get_estimated_value(x), 
                            reverse=self.desire == 'maximize_profit')

        chosen_items = []
        i = 0
        while budget >= 0 and i < len(sorted_items):
            item = sorted_items[i]
            if item.price <= budget:  # Select the item if it is affordable.
                chosen_items.append(item)
                budget -= item.price
            i += 1

        return chosen_items


    def get_plan_instruct(self, items: List[Item]):
        """
        Generate a plan instruction string for the LLM to create a bidding plan.

        Args:
            items (List[Item]): List of items available for bidding.

        Returns:
            str: The formatted plan instruction string for the LLM.

        Description:
            - This method generates a textual instruction for the LLM, incorporating 
            the bidder's budget, number of items, and goal (`desire`).
            - It uses the `INSTRUCT_PLAN_TEMPLATE` to format the instruction message.
        """
        self.items = items  # Set the items list for the bidder.
        plan_instruct = INSTRUCT_PLAN_TEMPLATE.format(
            bidder_name=self.name,
            budget=self.budget,
            item_num=len(items),
            items_info=self._get_items_value_str(items),
            desire_desc=DESIRE_DESC[self.desire],
            learning_statement='' if not self.enable_learning else _LEARNING_STATEMENT
        )
        return plan_instruct


    def init_plan(self, plan_instruct: str):
        """
        Initialize the bidding plan using LLM instructions.

        Args:
            plan_instruct (str): Instruction string for the LLM to generate a bidding plan.

        Returns:
            str: The generated bidding plan if applicable, else an empty string.

        Description:
            - For non-LLM bidders (e.g., 'rule'), the plan is created manually based on budget and items.
            - Otherwise, the LLM generates a plan based on the provided instruction.
            - Updates the `status_quo`, `plan_instruct`, and `cur_plan` attributes with the generated plan.
        """
        if 'rule' in self.model_name: 
            return ''  # Skip LLM processing for rule-based bidders.

        # Initialize the current status quo of the auction.
        self.status_quo = {
            'remaining_budget': self.budget,
            'total_profits': {bidder: 0 for bidder in self.all_bidders_status.keys()},
            'winning_bids': {bidder: {} for bidder in self.all_bidders_status.keys()},
        }

        # Handle cases where no plan is required.
        if self.plan_strategy == 'none':
            self.plan_instruct = ''
            self.cur_plan = ''
            return None

        # Prepare the LLM input messages.
        system_msg = {"role": "system", "content": self.system_message}
        plan_msg = {"role": "user", "content": plan_instruct}
        messages = [system_msg, plan_msg]
        
        # Call the LLM to generate a plan.
        result = self._run_llm_standalone(messages)
        
        if self.verbose:
            print(get_colored_text(plan_msg.content, 'red'))
            print(get_colored_text(result, 'green'))

        # Update internal states with the generated plan.
        self.dialogue_history += [
            plan_msg,
            {"role": "assistant", "content": result}
        ]
        self.llm_prompt_history.append({
            'messages': [{x.role: x.content} for x in messages],
            'result': result,
            'tag': 'plan_0'
        })
        self.cur_plan = result
        self.plan_instruct = plan_instruct

        # Track changes of the plan.
        self.changes_of_plan.append([
            f"{self.cur_item_id} (Initial)", 
            False, 
            json.dumps(extract_jsons_from_text(result)[-1]),
        ])
        
        if self.verbose:
            print(f"Plan: {self.name} ({self.model_name}) for {self._get_cur_item()}.")
        return result

    
    def get_rebid_instruct(self, auctioneer_msg: str):
        """
        Logs the auctioneer's message and returns it.
        
        Args:
            auctioneer_msg (str): The message received from the auctioneer.
        
        Returns:
            str: The auctioneer's message.
        
        Description:
            - Updates the dialogue history to include the received message.
        """
        self.dialogue_history += [
            {"role": "user", "content": auctioneer_msg},
            {"role": "assistant", "content": ''}
        ]
        return auctioneer_msg

    def get_bid_instruct(self, auctioneer_msg: str, bid_round: int):
        """
        Constructs a bidding instruction based on the current round and auctioneer's message.
        
        Args:
            auctioneer_msg (str): The message from the auctioneer.
            bid_round (int): The current bidding round number.
        
        Returns:
            str: The formatted bidding instruction for the LLM.
        
        Description:
            - Formats the auctioneer's message.
            - Adds status information if it is the first bidding round.
            - Updates the dialogue history with the instruction.
        """
        auctioneer_msg = auctioneer_msg.replace(self.name, f'You ({self.name})')
        
        bid_instruct = INSTRUCT_BID_TEMPLATE.format(
            auctioneer_msg=auctioneer_msg, 
            bidder_name=self.name,
            cur_item=self._get_cur_item(),
            estimated_value=self._get_estimated_value(self._get_cur_item()),
            desire_desc=DESIRE_DESC[self.desire],
            learning_statement='' if not self.enable_learning else _LEARNING_STATEMENT
        )

        # For the first round, add the status quo information.
        if bid_round == 0:
            if self.plan_strategy in ['static', 'none']:
                bid_instruct = f"""The status quo of this auction so far is:\n"{json.dumps(self.status_quo, indent=4)}"\n\n{bid_instruct}\n---\n"""
        else:
            bid_instruct = f'Now, the auctioneer says: "{auctioneer_msg}"'
        
        self.dialogue_history += [
            {"role": "user", "content": auctioneer_msg},
            {"role": "assistant", "content": ''}
        ]
        return bid_instruct

    def bid_rule(self, cur_bid: int, min_markup_pct: float = 0.1):
        """
        Defines a basic rule-based bidding strategy.
        
        Args:
            cur_bid (int): The current highest bid for the item.
            min_markup_pct (float): Minimum percentage increase for the next bid.
        
        Returns:
            int: The next bid value or -1 if the bidder cannot afford to continue.
        
        Description:
            - If the current bid is zero or below, set the bid to the starting price of the item.
            - Otherwise, increase the bid by `min_markup_pct` times the item's starting price.
            - Update the dialogue history based on the decision.
        """
        cur_item = self._get_cur_item()
        
        # Calculate the next bid based on the current bid and markup.
        if cur_bid <= 0:
            next_bid = cur_item.price
        else:
            next_bid = cur_bid + min_markup_pct * cur_item.price

        # Ensure the bid is within the budget and the maximum bid count is not exceeded.
        if self.budget - next_bid >= 0 and self.rule_bid_cnt < self.max_bid_cnt:
            msg = int(next_bid)
            self.rule_bid_cnt += 1
        else:
            msg = -1  # Indicates withdrawal from the auction.

        # Update the dialogue history with the decision.
        content = f'The current highest bid for {cur_item.name} is ${cur_bid}. '
        content += "I'm out!" if msg < 0 else f"I bid ${msg}! (Rule generated)"
        self.dialogue_history += [
            {"role": "user", "content": ''},
            {"role": "assistant", "content": content}
        ]
        
        return msg

    def bid(self, bid_instruct):
        """
        Constructs a bid based on the bidding instruction and current history.
        
        Args:
            bid_instruct (str): The instruction for constructing the bid.
        
        Returns:
            str: The generated bid response.
        
        Description:
            - Uses the LLM to generate a bid based on the system and user messages.
            - Updates dialogue and prompt history.
            - Tracks the total number of bids.
        """
        if self.model_name == 'rule':
            return ''  # Skip bid generation for rule-based bidders.
        
        bid_msg = {"role": "user", "content": bid_instruct}
        
        # Prepare LLM messages based on the current plan strategy.
        if self.plan_strategy == 'none':
            messages = [{"role": "system", "content": self.system_message}]
        else:
            messages = [{"role": "system", "content": self.system_message},
                        {"role": "user", "content": self.plan_instruct},
                        {"role": "assistant", "content": self.cur_plan}]
        
        self.bid_history += [bid_msg]
        messages += self.bid_history
        
        # Call the LLM with prepared messages to generate a bid.
        result = self._run_llm_standalone(messages)
        
        self.bid_history += [{"role": "assistant", "content": self.cur_plan}]
        self.dialogue_history += [
            {"role": "user", "content": ''},
            {"role": "assistant", "content": result}
        ]
        
        self.llm_prompt_history.append({
            'messages': [{x['role']: x['content']} for x in messages],
            'result': result,
            'tag': f'bid_{self.cur_item_id}'
        })
        
        if self.verbose:
            print(bid_instruct)
            print(result)
        
            print(f"Bid: {self.name} ({self.model_name}) for {self._get_cur_item()}.")
        self.total_bid_cnt += 1
        
        return result

    def get_summarize_instruct(self, bidding_history: str, hammer_msg: str, win_lose_msg: str):
        instruct = INSTRUCT_SUMMARIZE_TEMPLATE.format(
            cur_item=self._get_cur_item(), 
            bidding_history=bidding_history, 
            hammer_msg=hammer_msg.strip(), 
            win_lose_msg=win_lose_msg.strip(), 
            bidder_name=self.name,
            prev_status=self._status_json_to_text(self.status_quo),
        )
        return instruct

    def summarize(self, instruct_summarize: str):
        '''
        Update belief/status quo
        status_quo = summarize(system_message, bid_history, prev_status + instruct_summarize)
        '''
        self.budget_history.append(self.budget)
        self.profit_history.append(self.profit)
        
        if self.model_name == 'rule': 
            self.rule_bid_cnt = 0   # reset bid count for rule bidder
            return ''
        
        messages = [{"role": "system", "content": self.system_message}]
        # messages += self.bid_history
        summ_msg = {"role": "user", "content": instruct_summarize}
        messages.append(summ_msg)

        status_quo_text = self._run_llm_standalone(messages)
        
        self.dialogue_history += [summ_msg, {"role": "assistant", "content": status_quo_text}]
        self.bid_history += [summ_msg, {"role": "assistant", "content": status_quo_text}]
        
        self.llm_prompt_history.append({
            'messages': [{x.role: x.content} for x in messages],
            'result': status_quo_text,
            'tag': f'summarize_{self.cur_item_id}'
        })

        cnt = 0
        while cnt <= 3:
            sanity_msg = self._sanity_check_status_json(extract_jsons_from_text(status_quo_text)[-1])
            if sanity_msg == '':
                # pass sanity check then track beliefs
                consistency_msg = self._belief_tracking(status_quo_text)
            else:
                sanity_msg = f'- {sanity_msg}'
                consistency_msg = ''
                
            if sanity_msg != '' or (consistency_msg != '' and self.correct_belief):
                err_msg = f"As {self.name}, here are some error(s) of your summary of the status JSON:\n{sanity_msg.strip()}\n{consistency_msg.strip()}\n\nPlease revise the status JSON based on the errors. Don't apologize. Just give me the revised status JSON.".strip()
                
                # print(f"{self.name}: revising status quo for the {cnt} time:")
                # print(get_colored_text(err_msg, 'green'))
                # print(get_colored_text(status_quo_text, 'red'))
                
                messages += [{"role": "assistant", "content": status_quo_text}, 
                             {"role": "user", "content": err_msg}]
                status_quo_text = self._run_llm_standalone(messages)
                self.dialogue_history += [
                    {"role": "assistant", "content": status_quo_text}, 
                    {"role": "user", "content": err_msg}
                ]
                cnt += 1
            else:
                break
        
        self.status_quo = extract_jsons_from_text(status_quo_text)[-1]

        if self.verbose:
            print(get_colored_text(instruct_summarize, 'blue'))
            print(get_colored_text(status_quo_text, 'green'))
        
            print(f"Summarize: {self.name} ({self.model_name}) for {self._get_cur_item()}.")
        
        return status_quo_text
    
    def get_replan_instruct(self):
        instruct = INSTRUCT_REPLAN_TEMPLATE.format(
            status_quo=self._status_json_to_text(self.status_quo),
            remaining_items_info=self._get_items_value_str(self._get_remaining_items()),
            bidder_name=self.name,
            desire_desc=DESIRE_DESC[self.desire],
            learning_statement='' if not self.enable_learning else _LEARNING_STATEMENT
        )
        return instruct

    def replan(self, instruct_replan: str):
        '''
        plan = replan(system_message, instruct_plan, prev_plan, status_quo + (learning) + instruct_replan)
        '''
        if self.model_name == 'rule': 
            self.withdraw = False
            self.cur_item_id += 1
            return ''
        
        if self.plan_strategy in ['none', 'static']:
            self.bid_history = []  # clear bid history
            self.cur_item_id += 1
            self.withdraw = False
            return 'Skip replanning for bidders with static or no plan.'
        
        replan_msg = {"role": "user", "content": instruct_replan}
        
        messages = [{"role": "system", "content": self.system_message},
                    {"role": "user", "content": self.plan_instruct},
                    {"role": "assistant", "content": self.cur_plan}]

        messages.append(replan_msg)

        result = self._run_llm_standalone(messages)
        
        new_plan_dict = extract_jsons_from_text(result)[-1]
        cnt = 0
        while len(new_plan_dict) == 0 and cnt < 2:
            err_msg = 'Your response does not contain a JSON-format priority list for items. Please revise your plan.'
            messages += [
                {"role": "assistant", "content": result},
                {"role": "user", "content": err_msg}
            ]
            result = self._run_llm_standalone(messages)
            new_plan_dict = extract_jsons_from_text(result)[-1]
            
            self.dialogue_history += [
                {"role": "assistant", "content": result},
                {"role": "user", "content": err_msg}
            ]
            cnt += 1
        
        old_plan_dict = extract_jsons_from_text(self.cur_plan)[-1]
        self.changes_of_plan.append([
            f"{self.cur_item_id + 1} ({self._get_cur_item('name')})", 
            self._change_of_plan(old_plan_dict, new_plan_dict),
            json.dumps(new_plan_dict)
        ])
    
        self.plan_instruct = instruct_replan
        self.cur_plan = result
        self.withdraw = False
        self.bid_history = []  # clear bid history
        self.cur_item_id += 1

        self.dialogue_history += [
            replan_msg,
            {"role": "assistant", "content": result},
        ]
        self.llm_prompt_history.append({
            'messages': [{x.role: x.content} for x in messages],
            'result': result,
            'tag': f'plan_{self.cur_item_id}'
        })
        
        if self.verbose:
            print(get_colored_text(instruct_replan, 'blue'))
            print(get_colored_text(result, 'green'))

            print(f"Replan: {self.name} ({self.model_name}).")
        return result
    
    def _change_of_plan(self, old_plan: dict, new_plan: dict):
        for k in new_plan:
            if new_plan[k] != old_plan.get(k, None):
                return True
        return False
        
    # *********** Belief Tracking and Sanity Check *********** #
    
    def bid_sanity_check(self, bid_price, prev_round_max_bid, min_markup_pct):
        # can't bid more than budget or less than previous highest bid
        if bid_price < 0:
            msg = None
        else:
            min_bid_increase = int(min_markup_pct * self._get_cur_item('price'))
            if bid_price > self.budget:
                msg = f"you don't have insufficient budget (${self.budget} left)"
            elif bid_price < self._get_cur_item('price'):
                msg = f"your bid is lower than the starting bid (${self._get_cur_item('price')})"
            elif bid_price < prev_round_max_bid + min_bid_increase:
                msg = f"you must advance previous highest bid (${prev_round_max_bid}) by at least ${min_bid_increase} ({int(100 * min_markup_pct)}%)."
            else:
                msg = None
        return msg

    def rebid_for_failure(self, fail_instruct: str):
        result = self.bid(fail_instruct)
        self.failed_bid_cnt += 1
        return result
    
    def _sanity_check_status_json(self, data: dict):
        if data == {}:
            return "Error: No parsible JSON in your response. Possibly due to missing a closing curly bracket '}', or unpasible values (e.g., 'profit': 1000 + 400, instead of 'profit': 1400)."

        # Check if all expected top-level keys are present
        expected_keys = ["remaining_budget", "total_profits", "winning_bids"]
        for key in expected_keys:
            if key not in data:
                return f"Error: Missing '{key}' field in the status JSON."

        # Check if "remaining_budget" is a number
        if not isinstance(data["remaining_budget"], (int, float)):
            return "Error: 'remaining_budget' should be a number, and only about your remaining budget."

        # Check if "total_profits" is a dictionary with numbers as values
        if not isinstance(data["total_profits"], dict):
            return "Error: 'total_profits' should be a dictionary of every bidder."
        for bidder, profit in data["total_profits"].items():
            if not isinstance(profit, (int, float)):
                return f"Error: Profit for {bidder} should be a number."

        # Check if "winning_bids" is a dictionary and that each bidder's entry is a dictionary with numbers
        if not isinstance(data["winning_bids"], dict):
            return "Error: 'winning_bids' should be a dictionary."
        for bidder, bids in data["winning_bids"].items():
            if not isinstance(bids, dict):
                return f"Error: Bids for {bidder} should be a dictionary."
            for item, amount in bids.items():
                if not isinstance(amount, (int, float)):
                    return f"Error: Amount for {item} under {bidder} should be a number."

        # If everything is fine
        return ""
    
    def _status_json_to_text(self, data: dict):
        if 'rule' in self.model_name: return ''
        
        # Extract and format remaining budget
        structured_text = f"* Remaining Budget: ${data.get('remaining_budget', 'unknown')}\n\n"
        
        # Extract and format total profits for each bidder
        structured_text += "* Total Profits:\n"
        if data.get('total_profits'):
            for bidder, profit in data['total_profits'].items():
                structured_text += f"  * {bidder}: ${profit}\n"
        
        # Extract and list the winning bids for each item by each bidder
        structured_text += "\n* Winning Bids:\n"
        if data.get('winning_bids'):
            for bidder, bids in data['winning_bids'].items():
                structured_text += f"  * {bidder}:\n"
                if bids:
                    for item, amount in bids.items():
                        structured_text += f"    * {item}: ${amount}\n"
                else:
                    structured_text += f"    * No winning bids\n"
        
        return structured_text.strip()

    def _belief_tracking(self, status_text: str):
        '''
        Parse status quo and check if the belief is correct.
        '''
        belief_json = extract_jsons_from_text(status_text)[-1]
        # {"remaining_budget": 8000, "total_profits": {"Bidder 1": 1300, "Bidder 2": 1800, "Bidder 3": 0}, "winning_bids": {"Bidder 1": {"Item 2": 1200, "Item 3": 1000}, "Bidder 2": {"Item 1": 2000}, "Bidder 3": {}}}
        budget_belief = belief_json['remaining_budget']
        profits_belief = belief_json['total_profits']
        winning_bids = belief_json['winning_bids']

        msg = ''
        # track belief of budget
        self.total_self_belief_cnt += 1
        if budget_belief != self.budget:
            msg += f'- Your belief of budget is wrong: you have ${self.budget} left, but you think you have ${budget_belief} left.\n'
            self.self_belief_error_cnt += 1
            self.budget_error_history.append([
                self._get_cur_item('name'),
                budget_belief,
                self.budget,
            ])
        
        # track belief of profits
        for bidder_name, profit in profits_belief.items():
            if self.all_bidders_status.get(bidder_name) is None:
                # due to a potentially unreasonable parsing
                continue
            
            if self.name in bidder_name: 
                bidder_name = self.name
                self.total_self_belief_cnt += 1
            else:
                self.total_other_belief_cnt += 1
            
            real_profit = self.all_bidders_status[bidder_name]['profit']
            
            if profit != real_profit:
                if self.name == bidder_name:
                    self.self_belief_error_cnt += 1
                else:
                    self.other_belief_error_cnt += 1

                msg += f'- Your belief of total profit of {bidder_name} is wrong: {bidder_name} has earned ${real_profit} so far, but you think {bidder_name} has earned ${profit}.\n'

                # add to history
                self.profit_error_history.append([
                    f"{bidder_name} ({self._get_cur_item('name')})",
                    profit,
                    real_profit
                ])

        # track belief of winning bids
        for bidder_name, items_won_dict in winning_bids.items():
            if self.all_bidders_status.get(bidder_name) is None:
                # due to a potentially unreasonable parsing
                continue

            real_items_won = self.all_bidders_status[bidder_name]['items_won']
            # items_won = [(item, bid_price), ...)]
            
            items_won_list = list(items_won_dict.keys())
            real_items_won_list = [str(x) for x, _ in real_items_won]
            
            if self.name in bidder_name:
                self.total_self_belief_cnt += 1
            else:
                self.total_other_belief_cnt += 1
            
            if not item_list_equal(items_won_list, real_items_won_list):
                if bidder_name == self.name:
                    self.self_belief_error_cnt += 1
                    _bidder_name = f'you'
                else:
                    self.other_belief_error_cnt += 1
                    _bidder_name = bidder_name
                
                msg += f"- Your belief of winning items of {bidder_name} is wrong: {bidder_name} won {real_items_won}, but you think {bidder_name} won {items_won_dict}.\n"

                self.win_bid_error_history.append([
                    f"{_bidder_name} ({self._get_cur_item('name')})",
                    ', '.join(items_won_list),
                    ', '.join(real_items_won_list)
                ])
        
        return msg
    
    def win_bid(self, item: Item, bid: int):
        self.budget -= bid
        self.profit += item.true_value - bid
        self.items_won += [[item, bid]]
        msg = f"Congratuations! You won {item} at ${bid}."# Now you have ${self.budget} left. Your total profit so far is ${self.profit}."
        return msg
    
    def lose_bid(self, item: Item):
        return f"You lost {item}."# Now, you have ${self.budget} left. Your total profit so far is ${self.profit}."
    
    # set the profit information of other bidders
    def set_all_bidders_status(self, all_bidders_status: dict):
        self.all_bidders_status = all_bidders_status.copy()

    def set_withdraw(self, bid: int):
        if bid < 0:     # withdraw
            self.withdraw = True
        elif bid == 0:  # enable discount and bid again
            self.withdraw = False
        else:           # normal bid
            self.withdraw = False
            self.engagement_count += 1
            self.engagement_history[self._get_cur_item('name')] += 1
    
    # ****************** Logging ****************** #
    
    # def _parse_hedging(self, plan: str):   # deprecated
    #     prompt = PARSE_HEDGE_INSTRUCTION.format(
    #         item_name=self._get_cur_item(), 
    #         plan=plan)
        
    #     with get_openai_callback() as cb:
    #         llm = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=0)
    #         result = llm([HumanMessage(content=prompt)]).content
    #         self.openai_cost += cb.total_cost
    #         # parse a number, which could be a digit
    #         hedge_percent = re.findall(r'\d+\.?\d*%', result)
    #         if len(hedge_percent) > 0:
    #             hedge_percent = hedge_percent[0].replace('%', '')
    #         else:
    #             hedge_percent = 0
    #     return float(hedge_percent)
    
    def profit_report(self):
        '''
        Personal profit report at the end of an auction.
        '''
        msg = f"* {self.name}, starting with ${self.original_budget}, has won {len(self.items_won)} items in this auction, with a total profit of ${self.profit}.:\n"
        profit = 0
        for item, bid in self.items_won:
            profit += item.true_value - bid
            msg += f"  * Won {item} at ${bid} over ${item.price}, with a true value of ${item.true_value}.\n"
        return msg.strip()
    
    def to_monitors(self, as_json=False):
        # budget, profit, items_won, tokens
        if len(self.items_won) == 0 and not as_json: 
            items_won = [['', 0, 0]]
        else:
            items_won = []
            for item, bid in self.items_won:
                items_won.append([str(item), bid, item.true_value])
        
        profit_error_history = self.profit_error_history if self.profit_error_history != [] or as_json else [['', '', '']]
        win_bid_error_history = self.win_bid_error_history if self.win_bid_error_history != [] or as_json else [['', '', '']]
        budget_error_history = self.budget_error_history if self.budget_error_history != [] or as_json else [['', '']]
        changes_of_plan = self.changes_of_plan if self.changes_of_plan != [] or as_json else [['', '', '']]
        
        if as_json:
            return {
                'auction_hash': self.auction_hash,
                'bidder_name': self.name,
                'model_name': self.model_name,
                'desire': self.desire,
                'plan_strategy': self.plan_strategy,
                'overestimate_percent': self.overestimate_percent,
                'temperature': self.temperature,
                'correct_belief': self.correct_belief,
                'enable_learning': self.enable_learning,
                'budget': self.original_budget,
                'money_left': self.budget,
                'profit': self.profit,
                'items_won': items_won,
                'tokens_used': self.llm_token_count,
                'openai_cost': round(self.openai_cost, 2),
                'failed_bid_cnt': self.failed_bid_cnt,
                'self_belief_error_cnt': self.self_belief_error_cnt,
                'other_belief_error_cnt': self.other_belief_error_cnt,
                'failed_bid_rate': round(self.failed_bid_cnt / (self.total_bid_cnt+1e-8), 2),
                'self_error_rate': round(self.self_belief_error_cnt / (self.total_self_belief_cnt+1e-8), 2),
                'other_error_rate': round(self.other_belief_error_cnt / (self.total_other_belief_cnt+1e-8), 2),
                'engagement_count': self.engagement_count,
                'engagement_history': self.engagement_history,
                'changes_of_plan': changes_of_plan,
                'budget_error_history': budget_error_history,
                'profit_error_history': profit_error_history,
                'win_bid_error_history': win_bid_error_history,
                'history': self.llm_prompt_history
            }
        else:
            return [
                self.budget, 
                self.profit, 
                items_won, 
                self.llm_token_count, 
                round(self.openai_cost, 2), 
                round(self.failed_bid_cnt / (self.total_bid_cnt+1e-8), 2), 
                round(self.self_belief_error_cnt / (self.total_self_belief_cnt+1e-8), 2), 
                round(self.other_belief_error_cnt / (self.total_other_belief_cnt+1e-8), 2), 
                self.engagement_count,
                draw_plot(f"{self.name} ({self.model_name})", self.budget_history, self.profit_history), 
                changes_of_plan,
                budget_error_history,
                profit_error_history, 
                win_bid_error_history
            ]

    def dialogue_to_chatbot(self):
        # chatbot: [[Human, AI], [], ...]
        # only dialogue will be sent to LLMs. chatbot is just for display.
        assert len(self.dialogue_history) % 2 == 0
        chatbot = []
        for i in range(0, len(self.dialogue_history), 2):
            # if exceeds the length of dialogue, append the last message
            human_msg = self.dialogue_history[i].content
            ai_msg = self.dialogue_history[i+1].content
            if ai_msg == '': ai_msg = None
            if human_msg == '': human_msg = None
            chatbot.append([human_msg, ai_msg])
        return chatbot


def draw_plot(title, hedge_list, profit_list):
    x1 = [str(i) for i in range(len(hedge_list))]
    x2 = [str(i) for i in range(len(profit_list))]
    y1 = hedge_list
    y2 = profit_list

    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Bidding Round')
    ax1.set_ylabel('Budget Left ($)', color=color)
    ax1.plot(x1, y1, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    for i, j in zip(x1, y1):
        ax1.text(i, j, str(j), color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Profit ($)', color=color)
    ax2.plot(x2, y2, color=color, marker='^')
    ax2.tick_params(axis='y', labelcolor=color)

    for i, j in zip(x2, y2):
        ax2.text(i, j, str(j), color=color)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    # fig.tight_layout()
    plt.title(title)

    return fig


def bidding_multithread(bidder_list: List[Bidder],  
                        instruction_list, 
                        func_type,
                        thread_num=5,
                        retry=1):
    '''
    auctioneer_msg: either a uniform message (str) or customed (list)
    '''
    assert func_type in ['plan', 'bid', 'summarize', 'replan']
    
    result_queue = queue.Queue()
    threads = []
    semaphore = threading.Semaphore(thread_num)

    def run_once(i: int, bidder: Bidder, auctioneer_msg: str):
        try:
            semaphore.acquire()
            if func_type == 'bid':
                
                result = bidder.bid(auctioneer_msg)
            elif func_type == 'summarize':
                result = bidder.summarize(auctioneer_msg)
            elif func_type == 'plan':
                result = bidder.init_plan(auctioneer_msg)
            elif func_type == 'replan':
                result = bidder.replan(auctioneer_msg)
            else:
                raise NotImplementedError(f'func_type {func_type} not implemented')
            result_queue.put((True, i, result))
        # except Exception as e:
        #     result_queue.put((False, i, str(trace_back(e))))
        finally:
            semaphore.release()

    if isinstance(instruction_list, str):
        instruction_list = [instruction_list] * len(bidder_list)
    
    for i, (bidder, msg) in enumerate(zip(bidder_list, instruction_list)):
        thread = threading.Thread(target=run_once, args=(i, bidder, msg))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join(timeout=600)
    
    results = [result_queue.get() for _ in range(len(bidder_list))]
    
    errors = []
    for success, id, result in results:
        if not success:
            errors.append((id, result))
    
    if errors:
        raise Exception(f"Error(s) in {func_type}:\n" + '\n'.join([f'{i}: {e}' for i, e in errors]))
    
    valid_results = [x[1:] for x in results if x[0]]
    valid_results.sort()
    
    return [x for _, x in valid_results]
    

def bidders_to_chatbots(bidder_list: List[Bidder], profit_report=False):
    if profit_report:   # usually at the end of an auction
        return [x.dialogue_to_chatbot() + [[x.profit_report(), None]] for x in bidder_list]
    else:
        return [x.dialogue_to_chatbot() for x in bidder_list]


def create_bidders(bidder_info_jsl, auction_hash):
    bidder_info_jsl = LoadJsonL(bidder_info_jsl)
    bidder_list = []
    for info in bidder_info_jsl:
        info['auction_hash'] = auction_hash
        bidder_list.append(Bidder.create(**info))
    return bidder_list