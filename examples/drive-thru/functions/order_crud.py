from langchain_core.tools import tool
import threading

order_book = {}

@tool
def create_order(order_id: str, order_details: list[dict]) -> str:
    """
    Create a new order.
    
    Parameters:
    - order_id (str): A string containing the order id.
    - order_details (list[dict]): A list of dictionaries containing the details of each item with item name as key and quantity, item_price, total_price as values.
    Returns:
    - str: A string indicating the success/failure of the order creation.
    """
    
    if not isinstance(order_id, str):
        raise ValueError("Order ID must be a string")
        return "Something went wrong. Please try again."
    
    if not isinstance(order_details, list):
        raise ValueError("Order details must be a list")
        return "Something went wrong. Please try again."
    
    if not all(isinstance(item, dict) for item in order_details):
        raise ValueError("Order details must be a list of dictionaries")
        return "Something went wrong. Please try again."
    
    def delayed_action():
        print(f"order details: {order_details}")
        order_book[order_id] = order_details
        print(f"order book: {order_book}")
        return "Order created successfully"
    
    timer = threading.Timer(10, delayed_action)
    timer.start()


@tool
def get_order(order_id: str):
    """
    Get an order from the order book.
    
    Parameters:
    - order_id (str): The ID of the order to get.
    
    Returns:
    - dict: A dictionary containing the order details.
    """
    if order_id not in order_book:
        # raise ValueError("Order ID not found")
        return {"error": "Order ID not found"}
    return order_book[order_id]


@tool
def cancel_order(order_id: str):
    """
    Delete an order from the order book.
    
    Parameters:
    - order_id (str): The ID of the order to delete.
    
    Returns:
    - bool: A boolean value indicating the success of the order deletion.
    """
    
    if not isinstance(order_id, str):
        # raise ValueError("Order ID must be a string")
        return "Something went wrong. Please try again."
    
    if order_id not in order_book:
        # raise ValueError("Order ID not found")
        return {"error": "Order ID not found"}
    
    del order_book[order_id]
    return "Order cancelled successfully"


@tool
def add_item_to_order(order_id: str, item_details: dict):
    """
    Add an item to an order.
    
    Parameters:
    - order_id (str): The ID of the order to add the item to.
    - item_details (dict): The details of the item to add.

    Returns:
    - str: A string indicating the success/failure of the item addition.
    """
    if order_id not in order_book:
        return {"error": "Order ID not found"}
    
    order_book[order_id].append(item_details)
    return "Item added to order successfully"


@tool
def delete_item_from_order(order_id: str, item_name: str):
    """
    Delete an item from an order.
    
    Parameters:
    - order_id (str): The ID of the order to delete the item from.
    - item_name (str): The name of the item to delete.
        
    Returns:
    - str: A string indicating the success/failure of the item deletion.
    """
    if order_id not in order_book:
        return {"error": "Order ID not found"}
    
    
    for item in order_book[order_id]:
        print(f"item: {item}")
        if list(item.keys())[0] == item_name:
            order_book[order_id].remove(item)
            break
    return "Item deleted from order successfully"

@tool
def update_quantity_of_item_in_order(order_id: str, item_name: str, quantity: int):
    """
    Update an item in an order.
    
    Parameters:
    - order_id (str): The ID of the order to update the item in.
    - item_name (str): The name of the item to update.
    - quantity (int): The quantity of the item to update.
    
    Returns:
    - str: A string indicating the success/failure of the item update.
    """
    
    if order_id not in order_book:
        return {"error": "Order ID not found"}
    
    for item in order_book[order_id]:
        if list(item.keys())[0] == item_name:
            item[item_name]["quantity"] = quantity
            break
    return "Item updated in order successfully"