"""
language_templates.py
Culturally tuned empathetic replies for each major language.
"""

TEMPLATES = {
    "en": {
        "positive": "That's wonderful to hear! Thank you for sharing your positive experience.",
        "neutral": "Thanks for your feedback. How can we assist you further?",
        "negative": "I'm truly sorry you had a poor experience. We care deeply about patient satisfaction. Would you like me to connect you to patient relations?",
    },
    "hi": {
        "positive": "यह सुनकर बहुत अच्छा लगा! अपना सकारात्मक अनुभव साझा करने के लिए धन्यवाद।",
        "neutral": "आपकी प्रतिक्रिया के लिए धन्यवाद। मैं आपकी और कैसे मदद कर सकता हूँ?",
        "negative": "हमें खेद है कि आपका अनुभव अच्छा नहीं रहा। हम आपकी मदद करना चाहते हैं — क्या मैं आपको पेशेंट रिलेशन्स टीम से जोड़ दूँ?",
    },
    "ta": {
        "positive": "அதை கேட்டு மகிழ்ச்சி! உங்கள் நல்ல அனுபவத்தை பகிர்ந்ததற்கு நன்றி.",
        "neutral": "உங்கள் கருத்துக்கு நன்றி. மேலும் எவ்வாறு உதவ முடியும்?",
        "negative": "உங்கள் அனுபவம் சிறப்பாக இல்லை என்பதை வருந்துகிறோம். உதவ எங்களைத் தொடர்பு கொள்ளவா?",
    },
    "te": {
        "positive": "దాన్ని విని చాలా ఆనందంగా ఉంది! మీ సానుకూల అనుభవాన్ని పంచుకున్నందుకు ధన్యవాదాలు.",
        "neutral": "మీ అభిప్రాయానికి ధన్యవాదాలు. నేను మీకు ఎలా సహాయం చేయగలను?",
        "negative": "మీ అనుభవం సంతోషకరంగా లేకపోవడం విచారకరం. రోగి సంబంధాల బృందంతో మిమ్మల్ని కలపనా?",
    },
}

def get_template(lang: str, sentiment: str) -> str:
    if lang in TEMPLATES and sentiment in TEMPLATES[lang]:
        return TEMPLATES[lang][sentiment]
    return TEMPLATES["en"].get(sentiment, "Thanks for your feedback.")
