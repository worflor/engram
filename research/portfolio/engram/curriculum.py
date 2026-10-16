"""
curriculum.py

500 Wikipedia articles across 25 domains of human knowledge.

each domain has 20 articles ordered from general to specific.
the interleaved() function yields them in round-robin order:
one article per domain per round, 25 articles per round, 20 rounds.

early rounds establish broad wells. later rounds deepen them.
the ordering within each domain follows the natural pedagogical
sequence: you learn "Cell" before "CRISPR", "Force" before "Dark matter".
"""

from __future__ import annotations

DOMAINS: dict[str, list[str]] = {

    "biology": [
        "Life", "Cell (biology)", "DNA", "Protein", "Evolution",
        "Natural selection", "Photosynthesis", "Genetics", "Organism", "Bacteria",
        "Virus", "Mitosis", "Gene", "Enzyme", "Neuron",
        "Immune system", "Metabolism", "Chromosome", "RNA", "Stem cell",
    ],

    "physics": [
        "Physics", "Classical mechanics", "Thermodynamics", "Electromagnetism", "Optics",
        "Quantum mechanics", "General relativity", "Nuclear physics", "Particle physics", "Wave",
        "Energy", "Force", "Gravity", "Speed of light", "Entropy",
        "Magnetic field", "Superconductivity", "Standard Model", "Dark matter", "Black hole",
    ],

    "chemistry": [
        "Chemistry", "Atom", "Chemical element", "Periodic table", "Chemical bond",
        "Molecule", "Chemical reaction", "Organic chemistry", "Acid", "Redox",
        "Polymer", "Catalysis", "Electrochemistry", "Crystal", "Biochemistry",
        "Chemical equilibrium", "Solvent", "Isomer", "Spectroscopy", "Nanotechnology",
    ],

    "mathematics": [
        "Mathematics", "Number", "Arithmetic", "Algebra", "Geometry",
        "Trigonometry", "Calculus", "Probability", "Statistics", "Linear algebra",
        "Number theory", "Topology", "Group (mathematics)", "Mathematical analysis", "Differential equation",
        "Complex number", "Graph theory", "Set theory", "Mathematical logic", "Pi",
    ],

    "computer_science": [
        "Computer science", "Algorithm", "Data structure", "Programming language", "Computer",
        "Software", "Computer network", "Operating system", "Database", "Artificial intelligence",
        "Machine learning", "Cryptography", "Internet", "World Wide Web", "Boolean algebra",
        "Information theory", "Computer vision", "Compiler", "Turing machine", "Quantum computing",
    ],

    "medicine": [
        "Medicine", "Disease", "Anatomy", "Surgery", "Pharmacology",
        "Pathology", "Epidemiology", "Antibiotic", "Vaccine", "Cancer",
        "Cardiovascular disease", "Diabetes", "Infection", "Public health", "Radiology",
        "Anesthesia", "Mental disorder", "Organ transplantation", "Pandemic", "Magnetic resonance imaging",
    ],

    "astronomy": [
        "Astronomy", "Star", "Planet", "Galaxy", "Sun",
        "Moon", "Solar System", "Milky Way", "Telescope", "Supernova",
        "Nebula", "Asteroid", "Comet", "Exoplanet", "Pulsar",
        "Quasar", "Big Bang", "Cosmic microwave background", "Hubble Space Telescope", "Dark energy",
    ],

    "earth_science": [
        "Earth", "Geology", "Atmosphere", "Ocean", "Climate",
        "Weather", "Earthquake", "Volcano", "Mineral", "Rock (geology)",
        "Plate tectonics", "Fossil", "Erosion", "Water cycle", "Glacier",
        "Soil", "Geomorphology", "Oceanography", "Paleontology", "Ice age",
    ],

    "history": [
        "History", "Civilization", "Ancient Egypt", "Ancient Greece", "Roman Empire",
        "Middle Ages", "Renaissance", "Age of Enlightenment", "French Revolution", "Industrial Revolution",
        "World War I", "World War II", "Cold War", "Colonialism", "Silk Road",
        "Byzantine Empire", "Ottoman Empire", "British Empire", "Slavery", "Decolonization",
    ],

    "philosophy": [
        "Philosophy", "Ethics", "Logic", "Metaphysics", "Epistemology",
        "Aesthetics", "Existentialism", "Utilitarianism", "Stoicism", "Empiricism",
        "Rationalism", "Plato", "Aristotle", "Immanuel Kant", "Socrates",
        "Political philosophy", "Philosophy of mind", "Free will", "Determinism", "Nihilism",
    ],

    "literature": [
        "Literature", "Poetry", "Novel", "Drama", "Fiction",
        "Folklore", "Epic poetry", "Tragedy", "Narrative", "Rhetoric",
        "William Shakespeare", "Homer", "Fable", "Romanticism", "Allegory",
        "Modernism", "Dante Alighieri", "Fairy tale", "Literary criticism", "Haiku",
    ],

    "music": [
        "Music", "Melody", "Harmony", "Rhythm", "Musical instrument",
        "Orchestra", "Classical music", "Jazz", "Blues", "Rock music",
        "Opera", "Symphony", "Sonata", "Piano", "Violin",
        "Guitar", "Johann Sebastian Bach", "Ludwig van Beethoven", "Musical notation", "Funk",
    ],

    "visual_arts": [
        "Art", "Painting", "Sculpture", "Drawing", "Printmaking",
        "Photography", "Color", "Perspective (graphical)", "Renaissance art", "Baroque",
        "Impressionism", "Cubism", "Abstract art", "Surrealism", "Pop art",
        "Fresco", "Leonardo da Vinci", "Pablo Picasso", "Pottery", "Mosaic",
    ],

    "economics": [
        "Economics", "Market (economics)", "Supply and demand", "Capitalism", "Socialism",
        "Gross domestic product", "Inflation", "Monetary policy", "Fiscal policy", "International trade",
        "Microeconomics", "Macroeconomics", "Game theory", "Adam Smith", "Stock market",
        "Central bank", "Poverty", "Globalization", "Labour economics", "Opportunity cost",
    ],

    "psychology": [
        "Psychology", "Cognition", "Memory", "Perception", "Emotion",
        "Learning", "Motivation", "Intelligence", "Personality psychology", "Developmental psychology",
        "Social psychology", "Clinical psychology", "Behaviorism", "Cognitive bias", "Consciousness",
        "Attention", "Sigmund Freud", "Neuroscience", "Attachment theory", "Placebo",
    ],

    "linguistics": [
        "Language", "Grammar", "Syntax", "Semantics", "Phonology",
        "Morphology (linguistics)", "Pragmatics", "Sociolinguistics", "Historical linguistics", "Writing system",
        "Alphabet", "Etymology", "Pidgin", "Creole language", "Sign language",
        "Translation", "Noam Chomsky", "Language acquisition", "Computational linguistics", "Endangered language",
    ],

    "politics": [
        "Politics", "Democracy", "Government", "Republic", "Monarchy",
        "Constitution", "Political party", "Election", "Sovereignty", "Separation of powers",
        "Federalism", "Communism", "Fascism", "Nationalism", "Human rights",
        "United Nations", "Diplomacy", "Ideology", "Anarchism", "Totalitarianism",
    ],

    "law": [
        "Law", "Justice", "Criminal law", "International law", "Contract",
        "Property law", "Tort", "Jurisprudence", "Court", "Judge",
        "Habeas corpus", "Due process", "Intellectual property", "Copyright", "Patent",
        "Trial", "Rule of law", "Common law", "Civil law (legal system)", "Legal person",
    ],

    "engineering": [
        "Engineering", "Structural engineering", "Mechanical engineering", "Electrical engineering", "Civil engineering",
        "Chemical engineering", "Bridge", "Dam", "Car", "Aircraft",
        "Robot", "Turbine", "Electric motor", "Concrete", "Steel",
        "Power station", "Telecommunications", "Hydraulics", "Sewage treatment", "3D printing",
    ],

    "ecology": [
        "Ecology", "Ecosystem", "Biodiversity", "Food chain", "Habitat",
        "Conservation biology", "Extinction", "Deforestation", "Climate change", "Pollution",
        "Coral reef", "Rainforest", "Wetland", "Endangered species", "Invasive species",
        "Symbiosis", "Predation", "Migration", "Biome", "Sustainability",
    ],

    "religion": [
        "Religion", "God", "Christianity", "Islam", "Hinduism",
        "Buddhism", "Judaism", "Sikhism", "Atheism", "Monotheism",
        "Prayer", "Ritual", "Theology", "Soul", "Afterlife",
        "Pilgrimage", "Religious text", "Mysticism", "Taoism", "Confucianism",
    ],

    "anthropology": [
        "Anthropology", "Culture", "Human", "Society", "Kinship",
        "Marriage", "Tribe", "Ethnography", "Social stratification", "Gender",
        "Archaeology", "Cultural relativism", "Totem", "Taboo", "Oral tradition",
        "Rite of passage", "Indigenous peoples", "Hunter-gatherer", "Caste", "Ethnocentrism",
    ],

    "geography": [
        "Geography", "Continent", "Mountain", "River", "Desert",
        "Island", "Latitude", "Population", "Urbanization", "Map",
        "Cartography", "Border", "Tropics", "Tundra", "Steppe",
        "Peninsula", "Archipelago", "Equator", "Mediterranean climate", "Time zone",
    ],

    "agriculture": [
        "Agriculture", "Farm", "Crop", "Livestock", "Irrigation",
        "Wheat", "Rice", "Maize", "Fertilizer", "Pesticide",
        "Organic farming", "Domestication", "Aquaculture", "Beekeeping", "Vineyard",
        "Food security", "Agroforestry", "Crop rotation", "Green Revolution", "Soil fertility",
    ],

    "technology": [
        "Technology", "Invention", "Electricity", "Telephone", "Television",
        "Radio", "Semiconductor", "Transistor", "Laser", "Nuclear power",
        "Solar energy", "Electric battery", "Optical fiber", "Global Positioning System", "Biotechnology",
        "Automation", "Satellite", "Rocket", "Unmanned aerial vehicle", "Blockchain",
    ],
}

DOMAIN_ORDER = list(DOMAINS.keys())
ROUNDS = len(next(iter(DOMAINS.values())))  # 20


def interleaved() -> list[tuple[str, str, int]]:
    """
    yield (title, domain, round_index) in round-robin order.

    round 0: first article from each of 25 domains (25 articles).
    round 1: second article from each of 25 domains. etc.
    20 rounds x 25 domains = 500 articles.
    """
    schedule = []
    for round_idx in range(ROUNDS):
        for domain in DOMAIN_ORDER:
            articles = DOMAINS[domain]
            if round_idx < len(articles):
                schedule.append((articles[round_idx], domain, round_idx))
    return schedule


def domain_count() -> int:
    return len(DOMAINS)


def total_articles() -> int:
    return sum(len(arts) for arts in DOMAINS.values())
