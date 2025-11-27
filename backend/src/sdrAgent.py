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


# Company FAQ Data - Zerodha (Indian Stock Broker)
COMPANY_FAQ = {
    "company_name": "Zerodha",
    "tagline": "India's largest stock broker",
    "description": "Zerodha is a technology-driven discount broker offering stock trading, mutual funds, bonds, and more with zero brokerage on equity delivery and direct mutual funds.",
    
    "faqs": [
        {
            "question": "What does Zerodha do?",
            "answer": "Zerodha is India's largest stock broker that provides a platform for trading stocks, mutual funds, bonds, commodities, and currencies. We focus on technology and transparency, offering trading with minimal costs."
        },
        {
            "question": "What are the pricing or charges?",
            "answer": "We charge zero brokerage on equity delivery trades and direct mutual funds. For intraday and F&O trading, it's flat ₹20 per executed order or 0.03% whichever is lower. Account opening is free, and AMC is ₹300 per year."
        },
        {
            "question": "Is there a free tier or trial?",
            "answer": "Account opening is completely free. You can open a Demat and trading account at zero cost. The annual maintenance charge of ₹300 is charged only from the second year onwards."
        },
        {
            "question": "Who is Zerodha for?",
            "answer": "Zerodha is for anyone looking to invest or trade in Indian stock markets - from beginners starting their investment journey to active traders and seasoned investors. We serve over 1.5 crore clients across India."
        },
        {
            "question": "What platforms do you offer?",
            "answer": "We offer Kite, our flagship web and mobile trading platform, Coin for mutual funds, and Console for portfolio analytics. All platforms are designed to be fast, intuitive, and feature-rich."
        },
        {
            "question": "How do I get started?",
            "answer": "You can open an account online in under 10 minutes. You'll need your PAN card, Aadhaar, bank details, and a signature. The entire process is paperless and can be completed from your phone."
        },
        {
            "question": "What support do you provide?",
            "answer": "We offer 24/7 customer support through phone, email, and live chat. We also have extensive educational resources at Zerodha Varsity, which is completely free and covers everything from basics to advanced trading strategies."
        },
        {
            "question": "Is Zerodha safe and regulated?",
            "answer": "Yes, Zerodha is registered with SEBI and is a member of NSE, BSE, and MCX. We follow all regulatory requirements and client funds are held in separate accounts. We're also the first Indian broker to be profitable and debt-free."
        }
    ],
    
    "key_features": [
        "Zero brokerage on equity delivery",
        "Flat ₹20 per trade for intraday and F&O",
        "Free account opening",
        "Advanced trading platforms (Kite, Coin, Console)",
        "Educational resources through Zerodha Varsity",
        "No hidden charges"
    ]
}


class SDRAgent(Agent):
    def __init__(self, faq_data: dict) -> None:
        super().__init__(
            instructions=f"""You are a friendly and professional Sales Development Representative (SDR) for {faq_data['company_name']}.

Company Overview:
{faq_data['description']}

Your Role:
You help potential customers understand how {faq_data['company_name']} can solve their needs. You're knowledgeable, helpful, and focused on understanding the customer's requirements.

Conversation Flow:
1. Start with a warm greeting and ask what brought them here today
2. Listen to their needs and ask clarifying questions
3. Answer their questions using the company FAQ knowledge
4. Naturally collect lead information during the conversation:
   - Name
   - Company (if applicable)
   - Email address
   - Role/occupation
   - Primary use case (what they want to use this for)
   - Team size (if applicable - 1, 2-10, 11-50, 50+, or N/A for individual)
   - Timeline (now, within 1 month, within 3 months, just exploring)

5. When the user indicates they're done (says "that's all", "I'm done", "thanks bye", etc.), provide a brief summary and use the save_lead_summary tool

Guidelines:
- Be conversational and natural - you're speaking via voice
- Ask for information gradually, don't interrogate
- Weave lead qualification questions naturally into the conversation
- Use the answer_faq tool when the user asks product/pricing/company questions
- Keep responses concise and clear
- If you don't know something not in the FAQ, be honest and offer to connect them with the team
- Always confirm email addresses by spelling them out for accuracy

Remember: Build rapport first, understand their needs, then qualify the lead naturally!""",
        )
        
        self.faq_data = faq_data
        
        # Initialize lead state
        self.lead_state = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
            "conversation_notes": [],
            "timestamp": None
        }

    @function_tool
    async def answer_faq(
        self,
        context: RunContext,
        user_question: Annotated[str, "The user's question about the company, product, or pricing"]
    ):
        """Search the FAQ knowledge base and return relevant answers to user questions.
        
        Use this tool when the user asks about:
        - What the company does
        - Pricing and charges
        - Features and platforms
        - Who the product is for
        - How to get started
        - Support options
        - Any other company/product related questions
        
        Args:
            user_question: The user's question to search in the FAQ
        """
        
        # Simple keyword-based search through FAQ
        user_question_lower = user_question.lower()
        relevant_answers = []
        
        # Check each FAQ entry
        for faq in self.faq_data["faqs"]:
            question_lower = faq["question"].lower()
            answer = faq["answer"]
            
            # Simple keyword matching
            keywords = user_question_lower.split()
            if any(keyword in question_lower or keyword in answer.lower() for keyword in keywords if len(keyword) > 3):
                relevant_answers.append({
                    "question": faq["question"],
                    "answer": answer
                })
        
        if not relevant_answers:
            return f"I don't have specific information about that in my knowledge base, but I'd be happy to connect you with our team who can provide detailed answers. Could you tell me a bit more about what you're looking for?"
        
        # Return the most relevant answer(s)
        if len(relevant_answers) == 1:
            return relevant_answers[0]["answer"]
        else:
            # Combine multiple relevant answers
            response = "Here's what I can tell you: "
            for i, item in enumerate(relevant_answers[:2]):  # Limit to 2 answers for brevity
                response += item["answer"]
                if i < len(relevant_answers) - 1:
                    response += " "
            return response

    @function_tool
    async def update_lead_info(
        self,
        context: RunContext,
        field: Annotated[str, "The field to update: name, company, email, role, use_case, team_size, or timeline"],
        value: Annotated[str, "The value for this field"]
    ):
        """Update lead information as you collect it during the conversation.
        
        Use this tool to store lead details as the user provides them naturally in conversation.
        Don't ask for all fields at once - collect them gradually.
        
        Args:
            field: Which field to update (name, company, email, role, use_case, team_size, timeline)
            value: The value provided by the user
        """
        
        if field in self.lead_state:
            self.lead_state[field] = value
            logger.info(f"Updated lead info - {field}: {value}")
            return f"Got it, I've noted your {field}."
        else:
            return "I couldn't update that field."

    @function_tool
    async def save_lead_summary(
        self,
        context: RunContext,
        summary_notes: Annotated[str, "A brief summary of the conversation and the lead's interests/needs"]
    ):
        """Save the complete lead information and conversation summary when the call is ending.
        
        Use this tool when:
        - The user indicates they're done ("that's all", "I'm done", "thanks", "goodbye")
        - You've collected the key lead information
        - The conversation is naturally concluding
        
        Args:
            summary_notes: A brief summary of what the lead is interested in and their main needs
        """
        
        # Update timestamp and notes
        self.lead_state["timestamp"] = datetime.now().isoformat()
        self.lead_state["conversation_notes"].append(summary_notes)
        
        # Create leads directory if it doesn't exist
        leads_dir = Path("leads")
        leads_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_slug = (self.lead_state.get("name") or "unknown").lower().replace(" ", "_")
        filename = leads_dir / f"lead_{name_slug}_{timestamp}.json"
        
        # Save lead to file
        with open(filename, 'w') as f:
            json.dump(self.lead_state, f, indent=2)
        
        logger.info(f"Lead saved: {self.lead_state}")
        
        # Create verbal summary
        name = self.lead_state.get("name") or "the prospect"
        company = self.lead_state.get("company")
        use_case = self.lead_state.get("use_case")
        timeline = self.lead_state.get("timeline")
        
        summary = f"Perfect! I've captured all the details for {name}"
        if company:
            summary += f" from {company}"
        summary += ". "
        
        if use_case:
            summary += f"You're interested in {use_case}. "
        
        if timeline:
            summary += f"Timeline: {timeline}. "
        
        summary += "Our team will review your information and reach out shortly. Thanks for your time today!"
        
        return summary


def prewarm(proc: JobProcess):
    """Prewarm function to load resources before the agent starts."""
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["faq"] = COMPANY_FAQ
    logger.info(f"Prewarmed with FAQ data for {COMPANY_FAQ['company_name']}")


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Get FAQ data from prewarm
    faq_data = ctx.proc.userdata["faq"]

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

    # Start the session with our SDR agent
    await session.start(
        agent=SDRAgent(faq_data),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))