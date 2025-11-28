import logging
import json
from pathlib import Path
from typing import Annotated
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class FoodOrderingAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and helpful Food & Grocery Ordering Assistant for FreshMart, your local grocery and quick commerce store.

Your job is to take customer orders via voice conversation. Be warm, enthusiastic, and helpful!

When helping customers:
1. Greet them warmly and explain you can help them order groceries and prepared food
2. Understand what they want - they might ask for:
   - Specific items (e.g., "I need milk and bread")
   - Quantities ("two boxes of pasta")
   - Recipes/ingredients (e.g., "ingredients for a peanut butter sandwich" or "what I need for pasta dinner")
3. For recipe requests, use the add_recipe_to_cart tool - it automatically adds all needed items
4. For specific items, use the add_to_cart tool with the item name and quantity
5. Use available_items to search if customer asks about what's available
6. Show them their cart when they ask or after major additions
7. When they're done, use the place_order tool to save their order

Guidelines:
- Be conversational and natural - this is a voice conversation
- Ask clarifying questions one at a time
- When customers ask for recipe ingredients, intelligently map them to multiple items
- Common recipes: "peanut butter sandwich" (bread, peanut butter, jam), "pasta dinner" (pasta, sauce, olive oil), "breakfast" (bread, eggs, milk)
- Confirm items before adding to cart
- Keep a running total in your head
- When customer says "that's all", "I'm done", "place my order", or similar, finalize the order
- Always confirm the final order before saving
- Keep responses short and conversational - no lists or complex formatting""",
        )
        
        # Load catalog
        self.catalog = self._load_catalog()
        self.cart = []
        self.customer_name = None

    def _load_catalog(self):
        """Load catalog from JSON file"""
        try:
            with open("sharedData/catalog.json", "r") as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            logger.error("catalog.json not found!")
            return {"catalog": {}, "recipes": {}}

    @function_tool
    async def available_items(
        self,
        context: RunContext,
        search_term: Annotated[str, "Item name or category to search for"]
    ):
        """Search for available items in the catalog by name or category.
        
        Args:
            search_term: The item name or category to search for
        """
        search_term = search_term.lower()
        results = []
        
        # Search through all categories
        for category, items in self.catalog.get("catalog", {}).items():
            for item in items:
                if (search_term in item["name"].lower() or 
                    search_term in category.lower() or
                    any(tag in search_term for tag in item.get("tags", []))):
                    results.append({
                        "id": item["id"],
                        "name": item["name"],
                        "price": item["price"],
                        "category": item["category"],
                        "size": item.get("size", ""),
                        "brand": item.get("brand", "")
                    })
        
        if not results:
            return f"Sorry, I couldn't find '{search_term}' in our catalog."
        
        result_text = "Here's what I found:\n"
        for item in results[:5]:  # Limit to 5 results
            result_text += f"- {item['name']} ({item['brand']}) - ${item['price']} ({item['size']})\n"
        
        return result_text

    @function_tool
    async def add_to_cart(
        self,
        context: RunContext,
        item_name: Annotated[str, "The name of the item to add (e.g., 'milk', 'bread', 'pasta')"],
        quantity: Annotated[int, "Quantity to add"] = 1
    ):
        """Add an item to the customer's cart by name.
        
        Args:
            item_name: The name of the item to add
            quantity: How many to add (default is 1)
        """
        # Find the item in catalog by name
        item = None
        item_name_lower = item_name.lower()
        
        for category, items in self.catalog.get("catalog", {}).items():
            for i in items:
                if item_name_lower in i["name"].lower() or i["name"].lower() in item_name_lower:
                    item = i
                    break
            if item:
                break
        
        if not item:
            return f"Sorry, I couldn't find '{item_name}' in our catalog. Would you like me to search for similar items?"
        
        # Add to cart
        cart_item = {
            "name": item["name"],
            "item_id": item["id"],
            "quantity": quantity,
            "price": item["price"],
            "total": item["price"] * quantity
        }
        
        # Check if item already in cart
        existing = next((x for x in self.cart if x["item_id"] == item["id"]), None)
        if existing:
            existing["quantity"] += quantity
            existing["total"] = existing["price"] * existing["quantity"]
            return f"Updated! Now you have {existing['quantity']} of {item['name']} in your cart."
        else:
            self.cart.append(cart_item)
            return f"Added {quantity} {item['name']} to your cart for ${cart_item['total']:.2f}."

    @function_tool
    async def view_cart(self, context: RunContext):
        """Show the customer what's currently in their cart."""
        if not self.cart:
            return "Your cart is empty!"
        
        cart_summary = "Here's what's in your cart:\n"
        total = 0
        for item in self.cart:
            cart_summary += f"- {item['quantity']}x {item['name']} - ${item['total']:.2f}\n"
            total += item['total']
        
        cart_summary += f"\nTotal: ${total:.2f}"
        return cart_summary

    @function_tool
    async def add_recipe_to_cart(
        self,
        context: RunContext,
        recipe_name: Annotated[str, "Name of the recipe or dish (e.g., 'peanut butter sandwich', 'pasta dinner')"]
    ):
        """Add all items needed for a recipe to the cart automatically.
        
        Args:
            recipe_name: The name of the recipe
        """
        recipe_key = recipe_name.lower().replace(" ", "_")
        recipes = self.catalog.get("recipes", {})
        
        if recipe_key not in recipes:
            return f"I don't have a specific recipe for '{recipe_name}'. Try: peanut butter sandwich, pasta dinner, or breakfast."
        
        recipe = recipes[recipe_key]
        added_items = []
        
        for item_id in recipe["items"]:
            # Find and add the item
            for category, items in self.catalog.get("catalog", {}).items():
                for item in items:
                    if item["id"] == item_id:
                        cart_item = {
                            "name": item["name"],
                            "item_id": item["id"],
                            "quantity": 1,
                            "price": item["price"],
                            "total": item["price"]
                        }
                        
                        # Check if already in cart
                        existing = next((x for x in self.cart if x["item_id"] == item["id"]), None)
                        if existing:
                            existing["quantity"] += 1
                            existing["total"] = existing["price"] * existing["quantity"]
                        else:
                            self.cart.append(cart_item)
                        
                        added_items.append(item["name"])
                        break
        
        if added_items:
            return f"Perfect! I've added all ingredients for {recipe['name']}: {', '.join(added_items)}. Your cart has been updated!"
        else:
            return f"Couldn't add items for {recipe['name']}. Please try again."

    @function_tool
    async def place_order(
        self,
        context: RunContext,
        customer_name: Annotated[str, "Customer's name for the order"]
    ):
        """Place the order and save it to a JSON file.
        
        Args:
            customer_name: The customer's name for the order
        """
        if not self.cart:
            return "Your cart is empty! Please add items before placing an order."
        
        # Calculate total
        total = sum(item["total"] for item in self.cart)
        
        # Create order object
        order = {
            "customer_name": customer_name,
            "timestamp": datetime.now().isoformat(),
            "items": self.cart,
            "total": total
        }
        
        # Create orders directory if it doesn't exist
        orders_dir = Path("orders")
        orders_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = orders_dir / f"order_{customer_name.lower().replace(' ', '_')}_{timestamp}.json"
        
        # Save order to file
        with open(filename, 'w') as f:
            json.dump(order, f, indent=2)
        
        logger.info(f"Order saved: {filename}")
        
        # Clear cart for next order
        self.cart = []
        
        # Return confirmation
        item_count = sum(item["quantity"] for item in order["items"])
        return f"Perfect! Your order for {item_count} items totaling ${total:.2f} has been placed and saved. Thank you for shopping at FreshMart, {customer_name}!"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with our food ordering agent
    await session.start(
        agent=FoodOrderingAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))