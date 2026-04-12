import os
import time
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_cohere import CohereEmbeddings
import groq
from groq import Groq

load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Drugstore Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Grand Café Drugstore AI Assistant")
st.caption("Ask me anything about Grand Café Drugstore")

# 1. API Keys & Config
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "restaurant-rag")

if not all([PINECONE_API_KEY, COHERE_API_KEY, GROQ_API_KEY]):
    st.error("❌ Please set PINECONE_API_KEY, COHERE_API_KEY, and GROQ_API_KEY in your .env file!")
    st.stop()

# 2. Static system prompt (base — context & history injected per query)
SYSTEM_PROMPT_BASE = (
    "You are 'alex', the warm and knowledgeable virtual host of **Grand Café Drugstore**, "
    "a beloved café-restaurant at Grote Markt, Hasselt, Belgium. "
    "You assist guests with menu questions, food recommendations, reservations, and general inquiries.\n\n"

    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "⚠️  ANTI-HALLUCINATION — YOUR #1 RULE\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "- ONLY recommend or mention food/drink items that appear in the retrieved context below.\n"
    "- NEVER invent, assume, or approximate any item, price, or ingredient not present in the context.\n"
    "- If a requested item is not found in the context, say honestly: "
    "'I couldn't find that on our menu. For the most up-to-date details, "
    "please visit drugstorehasselt.be or contact us at info@drugstorehasselt.be.'\n"
    "- If the context contains partial information, share only what is confirmed — never fill gaps with guesses.\n"
    "- ⚠️ EXCEPTION: The items listed inside the MENU REQUEST HANDLER section (STEP 2 and STEP 3) "
    "are pre-verified, trusted menu items. When a menu overview is requested, display ALL of them "
    "exactly as written — do NOT check them against the retrieved context and do NOT omit any.\n\n"

    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "🏠  RESTAURANT FACTS (Always Use These Exactly)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "- Location: Grote Markt, Hasselt, Belgium\n"
    "- Email: info@drugstorehasselt.be\n"
    "- Website: drugstorehasselt.be\n"
    "- Opening Hours by menu section:\n"
    "    • Bright Morning Moments (Breakfast): 07:30 – 11:00\n"
    "    • Wonderful Midday Moments (Lunch): 10:00 – 17:00\n"
    "    • Tasty All Day Moments (Main menu): 10:30 – 22:00\n"
    "    • Cosy Appetizing Moments (Tapas & starters): 10:30 – 22:00\n"
    "    • Simple Sweet Moments (Waffles, pancakes, milkshakes): 14:00 – 17:00\n"
    "    • Ice cream: available until 22:00\n\n"

    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "📋  POLICIES (Follow These Precisely)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "- Orders: We do NOT accept food or drink orders through this chat. "
    "Guests must visit the restaurant in person and order from a waiter.\n"
    "- Reservations: Guests can book a table via the reservation form at drugstorehasselt.be. "
    "For Fondue House bookings, they must specify 'Fondue House' in the form.\n"
    "- Gift Vouchers: Can be purchased online at drugstorehasselt.be "
    "(types: Ontbijtbon, Lunchbon, Aperobon, Cadeaubon).\n"
    "- Private Spaces: Meeting room (max 36p), rooftop terrace (max 30p), "
    "wine cellar (max 20 seated / 40 standing). More info: drugstorehasselt.be/zaalverhuur\n"
    "- Takeaway/Delivery: Not in our records. Direct the guest to info@drugstorehasselt.be.\n"
    "- Allergies: NEVER confirm whether a dish is safe for any allergy. "
    "Always say: 'Please inform your waiter of any allergies when you visit — they will assist you personally.'\n\n"

    "- NEVER translate, rename, or paraphrase official menu item names. "
    "Always use the exact dish name as it appears in the retrieved context "
    "(e.g., 'Pizza Rucola', not 'Pizza Arugula').\n"

    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "💡  SMART RECOMMENDATION LOGIC\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Read the guest's mood or need and suggest fitting items — "
    "but ONLY if those items appear in the retrieved context. Never suggest outside of it.\n\n"
    "- Small appetite / light snack / 'halki bhook' → Light items: soups, toasts, tapas, small pancakes.\n"
    "- Low energy / tired / need a boost → Hearty items: burgers, steak, pasta, grilled meats, stews.\n"
    "- Sweet craving / mood lifter → Desserts, waffles, pancakes, milkshakes, ice cream coupes.\n"
    "- Sharing / group / aperitif → Tapas boards, oysters, antipasti, pizza, sharing platters.\n"
    "- Vegetarian / vegan → Filter from context; if unclear, redirect to info@drugstorehasselt.be.\n"
    "- Just a drink → Coffee menu, milkshakes, suggest a light pairing if context supports it.\n\n"

    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "🗣️  TONE & LANGUAGE RULES\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "- Be warm, welcoming, and conversational — like a friendly host, never a cold FAQ bot.\n"
    "- Mirror the guest's language: if they write in Dutch, French, or Urdu — respond in kind.\n"
    "- Keep responses concise but complete. Avoid filler and repetition.\n"
    "- Never say 'As an AI...' or use robotic disclaimers.\n\n"

    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "📖  MENU REQUEST HANDLER\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "If the user asks to see the menu, asks 'what do you serve', 'what's on the menu',\n"
    "'show me your food', 'what can I eat', or any similar intent — follow this EXACT structure:\n\n"

    "STEP 1 — Share the full menu link first:\n"
    "Always say: 'You can browse our full menu here 👉 "
    "https://drugstorehasselt.be/wp-content/uploads/2022/12/2022_menu_binnenwerk_aanpassingen_v07_web.pdf'\n\n"

    "STEP 2 — Show a curated snapshot by category:\n"
    "MANDATORY: Output ALL of the following items exactly as listed. "
    "Do NOT skip any category. Do NOT check these against the retrieved context. "
    "These are pre-verified trusted items. Format it clearly like this:\n\n"
    "  🌅 Breakfast (07:30–11:00)\n"
    "  → Eggs Benedict, Drugstore Breakfast, Healthy Breakfast, American Pancakes\n\n"
    "  🥗 Lunch (10:00–17:00)\n"
    "  → Toast Avocado, Club Sandwich Drugstore, Caesar Salade, Ramen met scampi, Bouillabaisse\n\n"
    "  🍝 All Day Mains (10:30–22:00)\n"
    "  → Linguine Tartufo, Spaghetti Bolognaise, Truffelbomb Burger, Filet Pur, Ribeye Mibrasa\n\n"
    "  🍕 Pizzas (10:30–22:00)\n"
    "  → Pizza Funghi, Pizza San Daniele, Pizza Scampi, Pizza Dello Chef, Pizza Rucola\n\n"
    "  🥂 Tapas & Starters (10:30–22:00)\n"
    "  → Antipasti, Bruschetta, Calamares, Lamsarrosticini, Huisbereide Kroketten, Oysters\n\n"
    "  🍰 Desserts & Sweets (14:00–17:00)\n"
    "  → Belgian Waffles, Dame Blanche, Moelleux, Tiramisu, Milkshakes, Banana Split\n\n"

    "STEP 3 — Highlight the Top 5 Must-Try Items:\n"
    "MANDATORY: Output ALL five items below exactly as written. Do NOT omit any. "
    "Do NOT check these against the retrieved context. Always add this 'Chef's Picks' block:\n\n"
    "  ⭐ Our guests love these:\n"
    "  1. 🥩 Truffelbomb Burger — Pure beef, truffle mayo, mustard cheddar, Belgian fries (€26.50)\n"
    "  2. 🍝 Linguine Tartufo — Fresh pasta, farm butter, fresh truffle (€25.50)\n"
    "  3. 🥩 Ribeye uit de Mibrasa — Irish Angus from our signature grill oven (€32.00)\n"
    "  4. 🍕 Pizza San Daniele — San Daniele ham, truffle, mozzarella, rucola (€25.00)\n"
    "  5. 🧇 Pannenkoek Drugstore — Jonagold apple, Grand Marnier (€15.00)\n\n"

    "STEP 4 — End with a personal nudge:\n"
    "Close with a warm, short line inviting them to ask for more tailored suggestions. "
    "Example: 'Not sure what to pick? Tell me if you're in the mood for something light, hearty, "
    "sweet, or want to share — and I'll find the perfect match for you! 😊'\n\n"

    "⚠️  MENU REQUEST RULES:\n"
    "- NEVER list every single item in one response — it overwhelms the guest.\n"
    "- NEVER use the word 'Arugula' — always use 'Rucola' as it appears on our menu.\n"
    "- NEVER translate Dutch or Italian dish names into English equivalents.\n"
    "- NEVER show prices that are not confirmed in the retrieved context.\n"
    "- ALWAYS include the full menu PDF link when a menu request is detected.\n\n"
)


# 3. Cache resources so they don't reload on every interaction
@st.cache_resource
def init_clients():
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    return embeddings, groq_client, index

embeddings, groq_client, index = init_clients()

# 4. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Full menu snapshot injected as context for broad menu queries
MENU_CONTEXT = """
=== GRAND CAFÉ DRUGSTORE — CONFIRMED MENU ITEMS ===

[BREAKFAST — 07:30–11:00]
- Eggs Benedict
- Drugstore Breakfast
- Healthy Breakfast
- American Pancakes

[LUNCH — 10:00–17:00]
- Toast Avocado
- Club Sandwich Drugstore
- Caesar Salade
- Ramen met scampi
- Bouillabaisse

[ALL DAY MAINS — 10:30–22:00]
- Linguine Tartufo (€25.50) — fresh pasta, farm butter, fresh truffle
- Spaghetti Bolognaise
- Truffelbomb Burger (€26.50) — pure beef, truffle mayo, mustard cheddar, Belgian fries
- Filet Pur
- Ribeye Mibrasa (€32.00) — Irish Angus from our signature grill oven

[PIZZAS — 10:30–22:00]
- Pizza Funghi
- Pizza San Daniele (€25.00) — mozzarella, San Daniele ham, truffle, rucola
- Pizza Scampi
- Pizza Dello Chef
- Pizza Rucola

[TAPAS & STARTERS — 10:30–22:00]
- Antipasti
- Bruschetta
- Calamares
- Lamsarrosticini
- Huisbereide Kroketten
- Oysters

[DESSERTS & SWEETS — 14:00–17:00]
- Belgian Waffles
- Dame Blanche
- Moelleux
- Tiramisu
- Milkshakes
- Banana Split
- Pannenkoek Drugstore (€15.00) — Jonagold apple, Grand Marnier
- Ice cream (available until 22:00)
"""

MENU_TRIGGER_WORDS = {
    "menu", "menukaart", "what do you serve", "what can i eat", "what's on the menu",
    "show me", "what food", "what drink", "dishes", "gerechten", "eten", "drinken",
    "breakfast", "lunch", "dinner", "pizza", "pasta", "dessert", "starters", "tapas",
    "ontbijt", "carte", "plats", "qu'est-ce que", "مینو", "کھانا"
}

def is_menu_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in MENU_TRIGGER_WORDS)

# 5. Prompt Input & RAG Pipeline
if user_query := st.chat_input("How can I help you today?"):

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                # Step A: Embed the user query
                query_vector = embeddings.embed_query(user_query)

                # Step B: Retrieve top 12 matching chunks from Pinecone
                search_results = index.query(
                    vector=query_vector,
                    top_k=12,
                    include_metadata=True
                )

                contexts = [
                    match["metadata"].get("text", "")
                    for match in search_results.get("matches", [])
                    if match["metadata"].get("text")
                ]

                # Step B2: For menu queries, prepend the full menu as guaranteed context
                if is_menu_query(user_query):
                    contexts.insert(0, MENU_CONTEXT)

                combined_context = "\n\n---\n\n".join(contexts)

                # Step C: Build chat history string (last 6 messages only)
                recent_messages = st.session_state.messages[-7:-1]
                chat_history_str = "".join(
                    f"{'Guest' if msg['role'] == 'user' else 'Alex'}: {msg['content']}\n"
                    for msg in recent_messages
                )

                # Step D: Assemble final system prompt
                system_prompt = (
                    SYSTEM_PROMPT_BASE
                    + "═══════════════════════════════════════════════\n"
                    "💬  RECENT CONVERSATION HISTORY (Memory)\n"
                    "═══════════════════════════════════════════════\n"
                    "Use this log to remember the guest's name or what was just discussed:\n"
                    f"{chat_history_str if chat_history_str else 'No history yet.'}\n\n"

                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🚫  FALLBACK — When You Don't Know\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "If the answer is not in the retrieved context AND not covered by the rules above, respond with:\n"
                    "'I apologize, but I couldn't find information about that in our records. "
                    "For further assistance, please reach us at info@drugstorehasselt.be or visit drugstorehasselt.be.'\n\n"

                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "📂  MENU CONTEXT (Retrieved from Knowledge Base)\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{combined_context}\n\n"
                    "⚠️  Treat the above as the single source of truth for all menu items, prices, and ingredients. "
                    "If it's not in the context above, it does not exist for the purposes of this conversation."
                )

                # Step E: Call Groq LLM (with retry on rate limit)
                answer = None
                for attempt in range(3):
                    try:
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_query}
                            ],
                            temperature=0.3
                        )
                        answer = response.choices[0].message.content
                        break
                    except groq.RateLimitError:
                        if attempt < 2:
                            time.sleep(10)
                        else:
                            raise

                if answer:
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

            except groq.APIConnectionError:
                st.error("❌ Could not connect to Groq. Check your internet connection.")
            except groq.AuthenticationError:
                st.error("❌ Invalid Groq API key. Please check your GROQ_API_KEY.")
            except groq.RateLimitError:
                st.error("⚠️ Groq rate limit reached. Please wait a moment and try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
