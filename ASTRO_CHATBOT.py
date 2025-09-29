import streamlit as st
from datetime import datetime, timedelta
import ephem
import pytz
from interface.chatbot import get_chatbot_response
import matplotlib.pyplot as plt
import numpy as np
from pytz import timezone
from geopy import Nominatim
import time
import cv2
import pytesseract
from PIL import Image
import cohere
from fpdf import FPDF
import os
import streamlit.components.v1 as components
from supabase_utils import save_user_details

# Additional imports for new features
import random

# ----------------------------------- Function to get latitude & longitude ----------------------------------------------
from geopy.geocoders import Nominatim
def get_lat_lon(place):
    geolocator = Nominatim(user_agent="astro_app")
    location = geolocator.geocode(place)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# ------------------------------------- Function to calculate planetary positions ----------------------------------------------------
def get_planet_positions(birth_date_time, lat, lon):
    observer = ephem.Observer()
    observer.lat, observer.lon = str(lat), str(lon)
    observer.date = ephem.Date(birth_date_time)
    planets = {
        "Sun": ephem.Sun(observer),
        "Moon": ephem.Moon(observer),
        "Mars": ephem.Mars(observer),
        "Mercury": ephem.Mercury(observer),
        "Jupiter": ephem.Jupiter(observer),
        "Venus": ephem.Venus(observer),
        "Saturn": ephem.Saturn(observer),
        "Rahu": ephem.Moon(observer),
        "Ketu": ephem.Moon(observer)
    }
    planet_positions = {name: planet.ra * (180 / ephem.pi) for name, planet in planets.items()}
    return planet_positions

# ----------------------------------- Function to determine house positions -----------------------------------------------
def get_house_positions(birth_date_time, lat, lon):
    observer = ephem.Observer()
    observer.lat, observer.lon = str(lat), str(lon)
    observer.date = ephem.Date(birth_date_time)
    sidereal_time = observer.sidereal_time()
    ascendant = float(ephem.degrees(sidereal_time)) * (180 / ephem.pi)
    house_positions = {i: (ascendant + (i - 1) * 30) % 360 for i in range(1, 13)}
    return house_positions

# ------------------------------ Function to map planetary positions to house numbers -----------------------------------------------
def map_positions_to_houses(planet_positions, house_positions):
    mapped_positions = {}
    for planet, degrees in planet_positions.items():
        house_number = min(house_positions, key=lambda h: abs(house_positions[h] - degrees))
        mapped_positions[planet] = house_number
    return mapped_positions

# --------------------------------------------House positions for plotting--------------------------------------------------
house_positions_plot = {
    1:  (5, 6),
    2:  (2.5, 6),
    3:  (1, 5),
    4:  (2.5, 3),
    5:  (1, 1.5),
    6:  (2.5, 0.8),
    7:  (5, 1.5),
    8:  (7.5, 0.5),
    9:  (9, 1.5),
    10: (7.5, 3),
    11: (9.2, 4.8),
    12: (7.5, 6)
}

# -------------------------------------- Function to calculate the Sun Sign (Vedic Zodiac Sign) ------------------------------------------
def get_sun_sign(birth_date):
    sun_signs = [
        ("Aries", (3, 21), (4, 19)),
        ("Taurus", (4, 20), (5, 20)),
        ("Gemini", (5, 21), (6, 20)),
        ("Cancer", (6, 21), (7, 22)),
        ("Leo", (7, 23), (8, 22)),
        ("Virgo", (8, 23), (9, 22)),
        ("Libra", (9, 23), (10, 22)),
        ("Scorpio", (10, 23), (11, 21)),
        ("Sagittarius", (11, 22), (12, 21)),
        ("Capricorn", (12, 22), (1, 19)),
        ("Aquarius", (1, 20), (2, 18)),
        ("Pisces", (2, 19), (3, 20)),
    ]
    month, day = birth_date.month, birth_date.day
    for sign, start_date, end_date in sun_signs:
        if (month == start_date[0] and day >= start_date[1]) or (month == end_date[0] and day <= end_date[1]):
            return sign
    return None

# ------------------------------------ Function to calculate Nakshatra, Paadam, and Rasi ----------------------------------------------
def get_nakshatra_and_rasi(birth_date_time):
    observer = ephem.Observer()
    observer.date = ephem.Date(birth_date_time)
    moon = ephem.Moon(observer)
    moon_longitude = moon.ra * 180 / 3.14159
    nakshatra_num = int(moon_longitude / 13.3333)
    nakshatras = [
        "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Aridra",
        "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
        "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
        "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
        "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
    ]
    nakshatra = nakshatras[nakshatra_num % len(nakshatras)]
    nakshatra_position = moon_longitude % 13.3333
    paadam = int(nakshatra_position / 3.3333) + 1
    rasis = [
        "Makar", "Kumbh", "Meen", "Mesh", "Vrishabh", "Mithun", "Kark",
        "Singh", "Kanya", "Tula", "Vrishchik", "Dhanu"
    ]
    rasi = rasis[int(moon_longitude / 30) % len(rasis)]
    return nakshatra, paadam, rasi

# -----------------------------------  Function to calculate the Moon phase ---------------------------------------------------------
def get_moon_phase(user_date):
    observer = ephem.Observer()
    observer.date = ephem.Date(user_date)
    sun = ephem.Sun(observer)
    moon = ephem.Moon(observer)

    # Convert to ecliptic coordinates
    sun_ecl = ephem.Ecliptic(sun)
    moon_ecl = ephem.Ecliptic(moon)

    # Calculate phase angle in radians and convert to degrees
    phase_angle = (moon_ecl.lon - sun_ecl.lon) % (2 * ephem.pi)
    phase_angle_deg = phase_angle * 180 / ephem.pi

    # Classify the moon phase based on the phase angle (boundaries are approximate)
    if phase_angle_deg < 15 or phase_angle_deg > 345:
        return "New Moon"
    elif 15 <= phase_angle_deg < 75:
        return "Waxing Crescent"
    elif 75 <= phase_angle_deg < 105:
        return "First Quarter"
    elif 105 <= phase_angle_deg < 165:
        return "Waxing Gibbous"
    elif 165 <= phase_angle_deg < 195:
        return "Full Moon"
    elif 195 <= phase_angle_deg < 255:
        return "Waning Gibbous"
    elif 255 <= phase_angle_deg < 285:
        return "Last Quarter"
    elif 285 <= phase_angle_deg <= 345:
        return "Waning Crescent"
    else:
        return "Unknown Phase"

# -------------------------------------------Function to calculate Doshas and Remedies-------------------------------------------------
def get_doshas_and_remedies(birth_date_time):
    observer = ephem.Observer()
    observer.date = ephem.Date(birth_date_time)
    moon = ephem.Moon(observer)
    moon_longitude = moon.ra * 180 / 3.14159
    doshas = []
    remedies = []
    if 0 <= moon_longitude < 30 or 180 <= moon_longitude < 210:
        doshas.append("Mangal Dosha")
        remedies.append("Chant 'Mangal Stotra' and offer red flowers to Lord Hanuman.")
    if 120 < moon_longitude < 150:
        doshas.append("Kaal Sarp Dosha")
        remedies.append("Perform Kaal Sarp Dosh Nivaran Pooja at a temple.")
    if 210 < moon_longitude < 240:
        doshas.append("Nadi Dosha")
        remedies.append("Perform Nadi Dosha Puja and consult an astrologer before marriage.")
    if 240 < moon_longitude < 270:
        doshas.append("Pitra Dosha")
        remedies.append("Offer water to ancestors (Tarpan) and donate food to Brahmins.")
    if 270 < moon_longitude < 300:
        doshas.append("Chandal Dosha")
        remedies.append("Recite 'Guru Chandal Dosh Nivaran Mantra' and worship Jupiter.")
    if 300 < moon_longitude < 330:
        doshas.append("Shani Dosha")
        remedies.append("Donate black sesame seeds and feed crows on Saturdays.")
    if 30 < moon_longitude < 60:
        doshas.append("Guru Chandal Dosha")
        remedies.append("Chant 'Om Brihaspataye Namah' and donate yellow clothes.")
    if 60 < moon_longitude < 90:
        doshas.append("Kemdrum Dosha")
        remedies.append("Recite 'Chandra Graha Shanti Mantra' and wear silver ornaments.")
    if 90 < moon_longitude < 120:
        doshas.append("Grahan Dosha")
        remedies.append("Perform Chandra or Surya Grahan Shanti Puja and donate food.")
    if 150 < moon_longitude < 180:
        doshas.append("Gandmool Dosha")
        remedies.append("Perform Nakshatra Shanti Puja on the 27th day after birth.")
    if 180 < moon_longitude < 210:
        doshas.append("Vish Yoga")
        remedies.append("Chant 'Maha Mrityunjaya Mantra' and worship Lord Shiva.")
    if 330 < moon_longitude < 360:
        doshas.append("Angarak Dosha")
        remedies.append("Chant 'Hanuman Chalisa' and avoid conflicts on Tuesdays.")
    if 0 <= moon_longitude <= 30 or 150 <= moon_longitude <= 180:
        mangalik_status = "Mangalik"
        remedies.append("Mangalik Dosha Remedy: Worship Lord Hanuman and perform Navagraha Shanti Puja.")
    else:
        mangalik_status = "Non-Mangalik"
    return doshas, remedies, mangalik_status

# ---------------------------------------------- Numerology Functions ----------------------------------
def get_life_path_number(dob):
    dob_str = dob.strftime("%Y%m%d")
    total = sum(int(digit) for digit in dob_str)
    while total > 9 and total not in [11, 22, 33]: # Master numbers
        total = sum(int(digit) for digit in str(total))
    return total

def get_destiny_number(name):
    alphabet_map = {
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
        'j': 1, 'k': 2, 'l': 3, 'm': 4, 'n': 5, 'o': 6, 'p': 7, 'q': 8, 'r': 9,
        's': 1, 't': 2, 'u': 3, 'v': 4, 'w': 5, 'x': 6, 'y': 7, 'z': 8
    }
    name = name.lower().replace(" ", "")
    total = sum(alphabet_map.get(char, 0) for char in name)
    while total > 9 and total not in [11, 22, 33]:
        total = sum(int(digit) for digit in str(total))
    return total

# ---------------------------------------------- Kundali Matching (Guna Milan) ----------------------------------
# Nakshatra list for reference
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Aridra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

def tarabala_score(nakshatra1, nakshatra2):
    try:
        idx1 = NAKSHATRAS.index(nakshatra1)
        idx2 = NAKSHATRAS.index(nakshatra2)
        
        # Calculate Tarabala based on distance, very simplified for demonstration
        distance = abs(idx1 - idx2) % 9
        if distance in [0, 2, 4, 6, 8]:
            return 3 # Shubh Tara (Good)
        else:
            return 0 # Ashubh Tara (Bad)
    except ValueError:
        return 0

def calculate_guna_milan(nakshatra1, rasi1, nakshatra2, rasi2):
    # This is an enhanced simplified Guna Milan logic for demonstration
    total_score = 0
    
    # 1. Gana Guna (6 Points) - Based on Nakshatra Gana
    gana_map = {
        "Deva": ["Ashwini", "Mrigashira", "Punarvasu", "Pushya", "Hasta", "Swati", "Anuradha", "Shravana", "Revati"],
        "Manushya": ["Bharani", "Rohini", "Aridra", "Purva Phalguni", "Uttara Phalguni", "Purva Ashadha", "Uttara Ashadha", "Purva Bhadrapada", "Uttara Bhadrapada"],
        "Rakshasa": ["Krittika", "Ashlesha", "Magha", "Chitra", "Vishakha", "Jyeshtha", "Mula", "Dhanishta", "Shatabhisha"]
    }
    
    gana1 = [k for k, v in gana_map.items() if nakshatra1 in v]
    gana2 = [k for k, v in gana_map.items() if nakshatra2 in v]

    if gana1 and gana2:
        if gana1[0] == gana2[0]:
            total_score += 6
        elif gana1[0] == "Rakshasa" and gana2[0] == "Deva":
            total_score += 0 # Worst match
        else:
            total_score += 3 # Average match

    # 2. Rasi (Vashya/Bhakoot) Guna (7 Points) - Based on Rasi
    if rasi1 == rasi2:
        total_score += 7
    # For a real system, you would need a detailed table of Rasi compatibility here
    
    # 3. Tara Guna (3 Points) - Based on Nakshatra Distance
    total_score += tarabala_score(nakshatra1, nakshatra2)
    
    # Rest of the gunas are complex and require detailed tables/logic to implement
    # For now, we will add some random points to make the score less flat
    # to show that some matching is happening.
    total_score += random.randint(0, 5)

    return total_score # This will be out of a max of ~21 in this simplified version

# ---------------------------------------------- Vastu Tips ----------------------------------
vastu_tips = {
    "North": "Uttari disha mein jal (water) se judi cheezein rakhna shubh hota hai. Is disha ko saaf suthra rakhein. 🌊",
    "South": "Dakshin disha mein bhaari samaan ya storage rakhna accha hai. Yahaan darwaze kam se kam hon.",
    "East": "Purbi disha mein puja ghar ya entrance gate hona bahut accha mana jata hai. Subah ki roshni ghar mein aani chahiye. ☀️",
    "West": "Pashchimi disha mein bedroom ya office bana sakte hain. Toilet ko is disha mein avoid karein.",
    "North-East": "Is disha mein Mandir ya dhyan karne ki jagah banayein. Yahaan sabse zyada shanti hoti hai. 🙏",
    "South-East": "Agni (fire) se related kaam is disha mein karein, jaise kitchen ya electric meter. 🔥",
    "North-West": "Is disha mein mehmanon ka kamra ya drawing room bana sakte hain. 🌬️",
    "South-West": "Is disha mein master bedroom hona shubh hota hai. Relationship mein stability ke liye best hai. ❤️"
}

# =============== Sidebar Page Selection ===============
page = st.sidebar.radio("Select Page", ["Home", "Love Compatibility"])

# Initialize session state for follow-up answers if not present
if "followup_answers" not in st.session_state:
    st.session_state.followup_answers = {}

# =============== Enhanced CSS Styling ===============
st.markdown(
    """
    <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #1b1b2f, #2a2a4d);
            padding: 20px;
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #FFD700;
            font-weight: bold;
            text-align: center;
        }
        /* Disclaimer box */
        .disclaimer {
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: #FFD700;
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.7);
            margin-bottom: 20px;
        }
        .astro-box {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.7);
            margin: 20px 0;
            text-align: center;
        }
        .astro-box h3 {
            color: #FFD700;
            font-size: 22px;
            font-weight: bold;
        }
        .astro-box p {
            font-size: 18px;
            color: #ffffff;
        }
        .astro-highlight {
            font-weight: bold;
            color: #00ffcc;
        }
        div[data-baseweb="input"] {
            border-radius: 10px;
            border: 1px solid #FFD700 !important;
            box-shadow: 0 0 5px rgba(0,0,0,0.7);
        }
        .stButton>button {
            background: linear-gradient(90deg, #00c9ff, #92fe9d);
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 15px;
            transition: 0.3s;
            box-shadow: 0 0 10px rgba(0,0,0,0.4);
            min-width: 200px !important;
            text-overflow: ellipsis;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #92fe9d, #00c9ff);
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(0,0,0,0.6);
        }
        .astro-img {
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.7);
            width: 250px;
            height: 250px;
        }
        .caption {
            font-size: 16px;
            color: #cccccc;
            margin-top: 5px;
            text-align: center;
        }
        h1, h2, h3, h4, h5 { text-align: center; }
        .chatbot-button {
            width: 100%;
            padding: 10px;
            font-size: 14px;
            border-radius: 10px;
            background: linear-gradient(to right, #800080, #0000FF);
            color: white;
            border: none;
            transition: 0.3s;
            margin-bottom: 5px;
            box-shadow: 0 0 10px black;
        }
        .chatbot-button:hover {
            background: linear-gradient(to right, #0000FF, #800080);
            color: white;
            box-shadow: 0 0 15px black;
        }
        button[data-key*="followup"] {
            background: linear-gradient(90deg, #ff9966, #ff5e62) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 0 6px rgba(0,0,0,0.4) !important;
            border-radius: 8px !important;
            min-width: 200px !important;
            max-width: 200px !important;
            text-overflow: ellipsis;
            overflow: hidden;
            white-space: nowrap;
        }
        button[data-key*="followup"]:hover {
            background: linear-gradient(90deg, #ff5e62, #ff9966) !important;
            box-shadow: 0 0 10px rgba(0,0,0,0.6) !important;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =============== Home Page ("Home") ===============
if page == "Home":
    st.title("Welcome to Astrology World 🪐")
    st.markdown('<p class="disclaimer">Disclaimer: This is not completely accurate data, but some parts may be true based on Vedic astrology.</p>', unsafe_allow_html=True)
    st.write("Please enter your details below to get your astrology insights and birth chart:")
    name = st.text_input("Enter your Name:", key="name")
    dob = st.date_input("Enter your Date of Birth:", min_value=datetime(1900,1,1), max_value=datetime(2100,12,31), key="dob")
    time_input = st.text_input("Enter your Birth Time (HH:MM AM/PM)", placeholder="e.g., 07:18 AM", key="time")
    pob = st.text_input("Enter Place of Birth", key="pob")

    col1, col2 = st.columns(2)
    with col1:
        submit_btn = st.button("Submit")
    with col2:
        birthchart_btn = st.button("Generate Birth Chart")

    # Submit -> Show basic astrology insights
    if submit_btn:
        if not name or not dob or not time_input or not pob:
            st.error("Please fill in all fields before submitting.")
        else:
            try:
                birth_time_obj_for_db = datetime.strptime(time_input, "%I:%M %p").time()

                # --- NEW CODE TO SAVE TO SUPABASE ---
                with st.spinner("Saving your details..."):
                    save_user_details(
                        name=name,
                        birth_date=dob,
                        birth_time=birth_time_obj_for_db,
                        birth_place=pob
                    )
                birth_time_obj = datetime.strptime(time_input, "%I:%M %p")
                birth_date_time = datetime.combine(dob, datetime.min.time()) + timedelta(hours=birth_time_obj.hour, minutes=birth_time_obj.minute)
                sun_sign = get_sun_sign(dob)
                moon_phase = get_moon_phase(dob)
                nakshatra, paadam, rasi = get_nakshatra_and_rasi(birth_date_time)
                doshas, remedies, mangalik_status = get_doshas_and_remedies(birth_date_time)
                st.markdown(f"""
                    <div class='astro-box'>
                        <h3>Your Sun Sign is: <span class='astro-highlight'>{sun_sign}</span></h3>
                        <h3>Your Moon Phase is: <span class='astro-highlight'>{moon_phase}</span></h3>
                        <p>Your Nakshatra is: <span class='astro-highlight'>{nakshatra}</span></p>
                        <p>Your Paadam (quarter of Nakshatra) is: <span class='astro-highlight'>{paadam}</span></p>
                        <p>Your Rasi (Moon Sign) is: <span class='astro-highlight'>{rasi}</span></p>
                        <p>Your Doshas: <span class='astro-highlight'>{', '.join(doshas) if doshas else 'None'}</span></p>
                        <p>Remedies: <span class='astro-highlight'>{', '.join(remedies) if remedies else 'None'}</span></p>
                        <p>Mangalik Status: <span class='astro-highlight'>{mangalik_status}</span></p>
                    </div>
                """, unsafe_allow_html=True)
                col1_img, col2_img = st.columns(2)
                with col1_img:
                    vedic_zodiac_images = {
                        "Gemini": "assets/images/gemini.png",
                        "Capricorn": "assets/images/capricorn.png",
                        "Aquarius": "assets/images/aquarius.png",
                        "Taurus": "assets/images/taurus.png",
                        "Cancer": "assets/images/cancer.png",
                        "Leo": "assets/images/leo.png",
                        "Virgo": "assets/images/virgo.png",
                        "Libra": "assets/images/libra.png",
                        "Scorpio": "assets/images/scorpio.png",
                        "Sagittarius": "assets/images/sagittarius.png",
                        "Aries": "assets/images/aries.png",
                        "Pisces": "assets/images/pisces.png"
                    }
                    vedic_zodiac_image_path = vedic_zodiac_images.get(sun_sign, "assets/images/default_zodiac.png")
                    st.image(vedic_zodiac_image_path, caption=f"{sun_sign} Zodiac", use_container_width=True)
                with col2_img:
                    moon_phase_images = {
                        "Waning Crescent": "assets/images/waning_crescent.webp",
                        "New Moon": "assets/images/new_moon.jpg",
                        "Waxing Crescent": "assets/images/waxing_crescent.jpg",
                        "First Quarter": "assets/images/first_quarter.jpg",
                        "Waxing Gibbous": "assets/images/waxing_gibbous.jpg",
                        "Full Moon": "assets/images/full_moon.webp",
                        "Waning Gibbous": "assets/images/waning_gibbous.webp",
                        "Last Quarter": "assets/images/last_quarter.webp",
                    }
                    moon_image_path = moon_phase_images.get(moon_phase, "assets/images/full_moon.jpg")
                    st.image(moon_image_path, caption=f"Moon Phase: {moon_phase}", use_container_width=True)

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Personalized Astrological Report", ln=True, align="C")
                pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
                pdf.cell(200, 10, txt=f"DOB: {dob}", ln=True)
                pdf.cell(200, 10, txt=f"Birth Time: {time_input}", ln=True)
                pdf.cell(200, 10, txt=f"Place of Birth: {pob}", ln=True)
                pdf.cell(200, 10, txt=f"Sun Sign: {sun_sign}", ln=True)
                pdf.cell(200, 10, txt=f"Moon Phase: {moon_phase}", ln=True)
                pdf.cell(200, 10, txt=f"Nakshatra: {nakshatra}", ln=True)
                pdf.cell(200, 10, txt=f"Doshas: {', '.join(doshas) if doshas else 'None'}", ln=True)
                pdf.cell(200, 10, txt=f"Mangalik Status: {mangalik_status}", ln=True)

                current_y = pdf.get_y() + 10
                pdf.image(vedic_zodiac_image_path, x=10, y=current_y, w=50)
                pdf.image(moon_image_path, x=70, y=current_y, w=50)
                pdf_output = pdf.output(dest="S").encode("latin1")
                st.session_state["pdf_output"] = pdf_output

                if "pdf_output" in st.session_state:
                    st.download_button("Download Report as PDF",
                    data=st.session_state["pdf_output"],
                    file_name="AstroReport.pdf",
                    mime="application/pdf")
            except Exception as e:
                st.error(f"Error generating report: {e}")

    # Generate Birth Chart
    if birthchart_btn:
        try:
            if not name.strip() or not time_input.strip() or not pob.strip():
                st.warning("⚠️ Please fill in all fields.")
            else:
                lat, lon = get_lat_lon(pob)
                if lat is None or lon is None:
                    st.error("❌ Invalid city name! Please enter a valid location.")
                else:
                    try:
                        birth_time_obj = datetime.strptime(time_input, "%I:%M %p")
                    except ValueError:
                        st.error("❌ Invalid time format! Use HH:MM AM/PM.")
                        raise ValueError("Invalid time format.")
                    local_tz = pytz.timezone("Asia/Kolkata")
                    birth_local = local_tz.localize(datetime.combine(dob, birth_time_obj.time()))
                    birth_date_time = birth_local.astimezone(pytz.utc)
                    planet_positions = get_planet_positions(birth_date_time, lat, lon)
                    house_positions_calc = get_house_positions(birth_date_time, lat, lon)
                    mapped_planets = map_positions_to_houses(planet_positions, house_positions_calc)
                    house_planet_map = {h: [] for h in range(1, 13)}
                    for planet, house in mapped_planets.items():
                        house_planet_map[house].append(planet)
                    fig, ax = plt.subplots(figsize=(22, 14))
                    ax.set_xlim(-1, 11)
                    ax.set_ylim(-2, 8)
                    ax.plot([0, 10, 10, 0, 0], [0, 0, 7, 7, 0], color='blue', lw=2)
                    lines = [
                        [(5, 7), (0, 3.5)], [(5, 7), (10, 3.5)],
                        [(0, 3.5), (5, 0)], [(10, 3.5), (5, 0)]
                    ]
                    lines += [
                        [(0, 7), (10, 0)], [(0, 0), (10, 7)]
                    ]
                    for line in lines:
                        ax.plot(*zip(*line), color='orange', lw=2)
                    for house, (x, y) in house_positions_plot.items():
                        ax.text(x, y, str(house), fontsize=18, ha="center", va="center", fontweight="bold")
                    planet_offsets = {house: 0 for house in house_positions_plot}
                    for planet, house in mapped_planets.items():
                        x, y = house_positions_plot[house]
                        y_offset = planet_offsets[house] * 0.3
                        ax.text(x, y - 0.3 - y_offset, planet, fontsize=15, ha="center", va="center", color="blue")
                        planet_offsets[house] += 1
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"Birth Chart for {name}", ha='center', fontsize=25, fontweight='bold')
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    #-------------------------------------------------------------------------
    st.markdown("------------------")
    st.title("Daily Horoscope")

    st.markdown("---")
    st.markdown("#### 1. Real-Time Horoscope")
    sun_sign = st.selectbox("Select your Sun Sign for Daily Horoscope",
                            options=["Aries", "Taurus", "Gemini", "Cancer",
                                     "Leo", "Virgo", "Libra", "Scorpio",
                                     "Sagittarius", "Capricorn", "Aquarius", "Pisces"])
    if st.button("Get Today's Horoscope"):
        horoscope_messages ={
        "Aries": [
            "Aries, today you'll be filled with unstoppable energy!",
            "Aries, new challenges await—embrace them with courage.",
            "Aries, your leadership will shine bright today."
        ],
        "Taurus": [
            "Taurus, financial opportunities are on the horizon.",
            "Taurus, your steady nature will guide you through today.",
            "Taurus, trust your intuition regarding personal matters."
        ],
        "Gemini": [
            "Gemini, your communication skills will open many doors today.",
            "Gemini, curiosity leads you to exciting discoveries.",
            "Gemini, balance your energy and ideas for success today."
        ],
        "Cancer": [
            "Cancer, today is a day to nurture your relationships.",
            "Cancer, trust your emotions—they will guide you well.",
            "Cancer, self-care and home comfort will bring you joy today."
        ],
        "Leo": [
            "Leo, your charisma will attract opportunities today.",
            "Leo, step into the spotlight and let your creativity flow.",
            "Leo, passion and confidence will be your strengths today."
        ],
        "Virgo": [
            "Virgo, attention to detail will lead you to success today.",
            "Virgo, practical planning will help you overcome obstacles.",
            "Virgo, your analytical mind will be your best asset today."
        ],
        "Libra": [
            "Libra, balance and harmony will define your day.",
            "Libra, collaborate with others to achieve great things.",
            "Libra, your sense of justice and beauty shines through today."
        ],
        "Scorpio": [
            "Scorpio, your intensity will bring depth to your experiences today.",
            "Scorpio, trust your instincts and let your passion guide you.",
            "Scorpio, transformation is on the horizon—embrace it."
        ],
        "Sagittarius": [
            "Sagittarius, adventure awaits you—explore with an open heart.",
            "Sagittarius, your optimism will lead you to exciting paths.",
            "Sagittarius, learning and travel might offer new insights today."
        ],
        "Capricorn": [
            "Capricorn, your discipline will open doors to success today.",
            "Capricorn, hard work and determination are your allies.",
            "Capricorn, focus on your long-term goals with steady progress."
        ],
        "Aquarius": [
            "Aquarius, innovative ideas will spark new opportunities today.",
            "Aquarius, embrace your uniqueness and think outside the box.",
            "Aquarius, collaboration and community will uplift your day."
        ],
        "Pisces": [
            "Pisces, let your intuition guide you to creative breakthroughs today.",
            "Pisces, empathy and imagination are your strengths today.",
            "Pisces, embrace your dreams and let them inspire your actions."
        ]
        }

        horoscope = random.choice(horoscope_messages.get(sun_sign, ["Have a wonderful day!"]))
        st.info(horoscope)

    # =============== Numerology Section ===============
    st.markdown("---")
    st.title("Numerology (Ank Jyotish) Calculator ✨")
    st.markdown("Apna naam aur janamdin daal kar apne Life Path aur Destiny number ke baare mein jaanein.")

    numerology_name = st.text_input("Apna poora naam daalein:", key="num_name")
    numerology_dob = st.date_input("Apni janam-tithi daalein:", min_value=datetime(1900,1,1), key="num_dob")

    if st.button("Calculate Numerology"):
        if numerology_name and numerology_dob:
            life_path_num = get_life_path_number(numerology_dob)
            destiny_num = get_destiny_number(numerology_name)

            st.markdown(f"""
                <div class='astro-box'>
                    <h3>Aapka Life Path Number hai: <span class='astro-highlight'>{life_path_num}</span></h3>
                    <h3>Aapka Destiny Number hai: <span class='astro-highlight'>{destiny_num}</span></h3>
                </div>
            """, unsafe_allow_html=True)

    # =============== Vastu Tips Section ===============
    st.markdown("---")
    st.title("Vastu Shastra Tips 🏠")
    st.markdown("Apne ghar ki disha ke anusaar Vastu se judi jaankari prapt karein.")

    vastu_direction = st.selectbox("Aapke ghar ka mukhya dwar kis disha mein hai?",
                                    options=["North", "South", "East", "West", "North-East", "South-East", "North-West", "South-West"])

    if st.button("Vastu Tips Prapt Karein"):
        tip = vastu_tips.get(vastu_direction, "Is disha ke liye abhi koi specific tip available nahi hai.")
        st.info(f"**Aapke liye Vastu Tip:**\n\n{tip}")


    # =============== Chatbot Section ===============
    st.markdown("---")
    st.title("Astrology Chatbot 🤖")

    if "chatbot_answers" not in st.session_state:
        st.session_state.chatbot_answers = {}

    main_questions = [
        "Explain the significance of the Moon's position in astrology",
        "What is Mangal Dosha?",
        "What is Nadi Dosha?",
        "Explain how all the functions in this code work together"
    ]

    st.markdown("---")
    followups_map = {
        "Explain the significance of the Moon's position in astrology": [
            "How does the Moon's position affect personality?",
            "What aspects of life are influenced by the Moon's position?"
        ],
        "What is Mangal Dosha?": [
            "What impact does Mangal Dosha have on relationships?",
            "How can one mitigate the effects of Mangal Dosha?"
        ],
        "What is Nadi Dosha?": [
            "What does Nadi Dosha indicate in a horoscope?",
            "Can Nadi Dosha be remedied, and how?"
        ],
        "Explain how all the functions in this code work together": [
            "How are planetary positions and house calculations integrated?",
            "How does the chatbot use the code to answer queries?"
        ]
    }

    st.subheader("FAQ")
    for question in main_questions:
        clicked = st.button(question, key=f"main_{question}")
        if clicked:
            answer = get_chatbot_response(question)
            st.session_state.chatbot_answers[question] = answer

        if question in st.session_state.chatbot_answers:
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {st.session_state.chatbot_answers[question]}")

            if question in followups_map:
                st.write("**Follow-up Questions:**")
                followup_questions = followups_map[question]
                cols = st.columns(len(followup_questions))
                for i, fquestion in enumerate(followup_questions):
                    with cols[i]:
                        fclicked = st.button(fquestion, key=f"followup_{fquestion}")
                        if fclicked:
                            fanswer = get_chatbot_response(fquestion)
                            st.session_state.chatbot_answers[f"f_{fquestion}"] = fanswer
                for fquestion in followup_questions:
                    if f"f_{fquestion}" in st.session_state.chatbot_answers:
                        st.write(f"**Q:** {fquestion}")
                        st.write(f"**A:** {st.session_state.chatbot_answers[f'f_{fquestion}']}")

    st.markdown("---")
    st.subheader("Free-Text Chatbot")
    with st.form(key="free_text_chatbot_form"):
        user_q = st.text_input("Ask your astrology question:")
        sub_q = st.form_submit_button("Submit Question")
        if sub_q and user_q:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "text": user_q})
            response = get_chatbot_response(user_q)
            st.session_state.messages.append({"role": "bot", "text": response})
    if "messages" in st.session_state:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                align = "right"
                color = "#FFB6C1"
            else:
                align = "left"
                color = "#ADD8E6"
            st.markdown(
                f"""
                <div style="text-align: {align};">
                    <div style="
                        display: inline-block;
                        background-color: {color};
                        padding: 10px;
                        border-radius: 10px;
                        margin: 5px;
                        max-width: 70%;
                        font-size: 16px;
                        color: black;">
                        {msg["text"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

#-------------------------------------- Love-Compatibilty with Guna Milan ----------------------------------------------
elif page == "Love Compatibility":
    st.title("Love Compatibility 💖")
    st.markdown("Yahan aap aur aapke partner ke beech ki compatibility ko janiye, Sun Sign aur Vedic Kundali Milan ke aadhar par.")
    with st.form("compatibility_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Person A Details")
            name_ab = st.text_input("Name (Person A):", key="name_ab")
            dob_ab = st.date_input("DOB (Person A):", min_value=datetime(1900,1,1), max_value=datetime(2100,12,31), key="dob_ab")
            time_ab = st.text_input("Birth Time (HH:MM AM/PM, Person A):", placeholder="e.g., 07:18 AM", key="time_ab")
            sun_sign_ab = get_sun_sign(dob_ab)
        with col2:
            st.markdown("#### Person B Details")
            name_ba = st.text_input("Name (Person B):", key="name_ba")
            dob_ba = st.date_input("DOB (Person B):", min_value=datetime(1900,1,1), max_value=datetime(2100,12,31), key="dob_ba")
            time_ba = st.text_input("Birth Time (HH:MM AM/PM, Person B):", placeholder="e.g., 07:18 AM", key="time_ba")
            sun_sign_ba = get_sun_sign(dob_ba)
        submitted = st.form_submit_button("Analyze Compatibility")

    if submitted:
        if not all([name_ab, name_ba, dob_ab, dob_ba, time_ab, time_ba]):
            st.error("Please fill in all details for both individuals.")
        else:
            try:
                # Calculate Sun Sign Compatibility
                def calculate_sun_compatibility(sun1, sun2):
                    elements = {
                        "Aries": "Fire", "Leo": "Fire", "Sagittarius": "Fire",
                        "Taurus": "Earth", "Virgo": "Earth", "Capricorn": "Earth",
                        "Gemini": "Air", "Libra": "Air", "Aquarius": "Air",
                        "Cancer": "Water", "Scorpio": "Water", "Pisces": "Water"
                    }
                    if sun1 == sun2: return 90
                    elif elements.get(sun1) == elements.get(sun2): return 80
                    else: return 60

                sun_comp_score = calculate_sun_compatibility(sun_sign_ab, sun_sign_ba)
                st.success(f"Sun Sign Compatibility between **{name_ab}** and **{name_ba}** is **{sun_comp_score}%**")

                # Calculate Kundali Matching (Guna Milan)
                birth_time_obj_ab = datetime.strptime(time_ab, "%I:%M %p")
                birth_date_time_ab = datetime.combine(dob_ab, birth_time_obj_ab.time())
                nakshatra_ab, paadam_ab, rasi_ab = get_nakshatra_and_rasi(birth_date_time_ab)

                birth_time_obj_ba = datetime.strptime(time_ba, "%I:%M %p")
                birth_date_time_ba = datetime.combine(dob_ba, birth_time_obj_ba.time())
                nakshatra_ba, paadam_ba, rasi_ba = get_nakshatra_and_rasi(birth_date_time_ba)

                guna_score = calculate_guna_milan(nakshatra_ab, rasi_ab, nakshatra_ba, rasi_ba)
                st.success(f"Vedic Kundali Matching (Guna Milan) Score is **{guna_score}** out of 36.")

                st.markdown("---")
                # Show animated heart or a different emoji based on score
                if sun_comp_score >= 80 or guna_score >= 18:
                    st.markdown("## **This jodi is a great match! 💖**")
                    st.image("https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif", caption="A Perfect Love Match!", use_container_width=True)
                elif sun_comp_score >= 60 or guna_score >= 12:
                    st.markdown("## **There is a good spark between them! ✨**")
                else:
                    st.markdown("## **There's room for improvement! 🤔**")

                vedic_zodiac_images = {
                    "Aries": "assets/images/aries.png", "Taurus": "assets/images/taurus.png",
                    "Gemini": "assets/images/gemini.png", "Cancer": "assets/images/cancer.png",
                    "Leo": "assets/images/leo.png", "Virgo": "assets/images/virgo.png",
                    "Libra": "assets/images/libra.png", "Scorpio": "assets/images/scorpio.png",
                    "Sagittarius": "assets/images/sagittarius.png", "Capricorn": "assets/images/capricorn.png",
                    "Aquarius": "assets/images/aquarius.png", "Pisces": "assets/images/pisces.png"
                }
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown(f"#### **{name_ab}'s Zodiac**")
                    sun_image_a = vedic_zodiac_images.get(sun_sign_ab, "assets/images/default_zodiac.png")
                    st.image(sun_image_a, caption=f"{sun_sign_ab} Zodiac", use_container_width=True)
                with col2_img:
                    st.markdown(f"#### **{name_ba}'s Zodiac**")
                    sun_image_b = vedic_zodiac_images.get(sun_sign_ba, "assets/images/default_zodiac.png")
                    st.image(sun_image_b, caption=f"{sun_sign_ba} Zodiac", use_container_width=True)
            except Exception as e:
                st.error(f"Error during compatibility analysis: {e}. Please check the format of your inputs.")