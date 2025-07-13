

# ============================================================================
# MAPPINGS/DICTS
# ============================================================================

UNIT_TO_G = {
    "g": 1,
    "gram": 1,
    "grams": 1,

    "kg": 1000,

    "oz": 28.35,

    "ounce": 28.35,
    "ounces": 28.35,

    "lb": 453.6,
    "pound": 453.6,

    "tsp": 4,
    "teaspoon": 4,
    "teaspoons": 4,
    "tbsp": 12,
    "tablespoon": 12,
    "tablespoons": 12,

    "cup": 240,
    "cups": 240,
}


# Substitution mappings for common ingredients
KETO_SUBSTITUTIONS = {
    # Flours and grains
    "flour": ["almond flour", "coconut flour", "flaxseed meal"],
    "all purpose flour": ["almond flour", "coconut flour"],
    "wheat flour": ["almond flour", "coconut flour", "sunflower seed flour"],
    "bread": ["cloud bread", "keto bread", "lettuce wraps"],
    "breadcrumbs": ["crushed pork rinds", "almond flour", "parmesan cheese"],
    "pasta": ["zucchini noodles", "shirataki noodles", "spaghetti squash"],
    "rice": ["cauliflower rice", "shirataki rice"],

    # Sugars and sweeteners
    "sugar": ["erythritol", "stevia", "monk fruit sweetener"],
    "brown sugar": ["brown erythritol", "sukrin gold"],
    "honey": ["sugar-free maple syrup", "allulose syrup"],
    "maple syrup": ["sugar-free maple syrup", "keto maple syrup"],
    "corn syrup": ["sugar-free syrup", "allulose syrup"],

    # Dairy alternatives (for higher fat)
    "milk": ["heavy cream", "coconut milk", "almond milk"],
    "skim milk": ["heavy cream diluted with water", "unsweetened almond milk"],

    # Starchy vegetables
    "potato": ["cauliflower", "turnips", "radishes"],
    "sweet potato": ["pumpkin", "butternut squash (small amounts)"],
    "corn": ["baby corn (small amounts)", "yellow squash"],

    # Fruits
    "apple": ["small amounts of berries", "jicama"],
    "banana": ["avocado (for smoothies)", "small amounts of berries"],

    # Legumes
    "beans": ["green beans", "black soybeans"],
    "chickpeas": ["roasted almonds", "macadamia nuts"],

    # Condiments
    "ketchup": ["sugar-free ketchup", "tomato paste with stevia"],
    "bbq sauce": ["sugar-free bbq sauce", "dry rub seasonings"],
}

VEGAN_SUBSTITUTIONS = {
    # Dairy
    "milk": ["soy milk", "almond milk", "oat milk", "coconut milk"],
    "butter": ["vegan butter", "coconut oil", "olive oil"],
    "heavy cream": ["coconut cream", "cashew cream", "silken tofu blended"],
    "cream": ["coconut cream", "cashew cream", "soy cream"],
    "cheese": ["nutritional yeast", "vegan cheese", "cashew cheese"],
    "parmesan": ["nutritional yeast", "vegan parmesan"],
    "cream cheese": ["vegan cream cheese", "cashew cream cheese"],
    "sour cream": ["vegan sour cream", "coconut cream with lemon"],
    "yogurt": ["coconut yogurt", "soy yogurt", "almond yogurt"],

    # Eggs
    "egg": ["flax egg", "chia egg", "applesauce", "mashed banana"],
    "eggs": ["flax eggs", "chia eggs", "aquafaba", "tofu scramble"],

    # Meat and seafood
    "chicken": ["tofu", "tempeh", "seitan", "jackfruit"],
    "beef": ["mushrooms", "lentils", "walnut meat", "plant-based ground"],
    "ground beef": ["lentils", "mushrooms", "plant-based ground", "walnut meat"],
    "bacon": ["tempeh bacon", "mushroom bacon", "coconut bacon"],
    "sausage": ["vegan sausage", "seasoned tempeh", "mushroom sausage"],
    "fish": ["tofu", "hearts of palm", "banana blossom"],
    "shrimp": ["king oyster mushrooms", "hearts of palm"],

    # Other animal products
    "honey": ["maple syrup", "agave nectar", "date syrup"],
    "gelatin": ["agar agar", "carrageenan", "pectin"],
    "worcestershire sauce": ["soy sauce", "vegan worcestershire"],
    "fish sauce": ["soy sauce", "mushroom sauce", "seaweed"],

    # Baking
    "buttermilk": ["soy milk + vinegar", "almond milk + lemon juice"],
}



# ============================================================================
# LISTS/SETS
# ============================================================================


NON_KETO = list(set([
    "cooking spray",

    # High-carb fruits
    "apple", "banana", "orange", "grape", "kiwi", "mango", "peach",
    "strawberry", "strawberries", "pineapple", "apricot", "tangerine", "persimmon",
    "pomegranate", "prune", "papaya", "jackfruit", "watermelon", "cantaloupe",
    "honeydew", "plum", "cherry", "cherries", "blueberry", "blueberries",
    "raspberry", "raspberries", "blackberry", "blackberries", "cranberry", "cranberries",
    "fig", "figs", "date", "dates", "raisin", "raisins",

    # Grains and grain products
    "white rice", "long grain rice", "brown rice", "wild rice", "jasmine rice", "basmati rice",
    "cornmeal", "corn", "all-purpose flour", "bread", "pasta", "couscous",
    "bulgur", "quinoa", "barley flour", "buckwheat flour", "barley",
    "durum wheat flour", "wheat flour", "whole-wheat flour", "wheat",
    "oat flour", "oatmeal", "oats", "rye flour", "semolina", "rye",
    "amaranth", "millet", "sorghum flour", "sorghum grain",
    "spelt flour", "teff grain", "triticale",
    "einkorn flour", "emmer grain", "fonio", "freekeh",
    "kamut flour", "farina", "polenta", "hominy", "grits",

    # Starchy vegetables
    "potato", "baking potato", "potato wedge", "potato slice",
    "russet potato", "sweet potato", "yam", "cassava",
    "taro", "lotus root", "water chestnut", "ube", "plantain",
    "parsnip", "rutabaga", "turnip", "beet", "beets", "beetroot",
    "carrot", "carrots", "corn kernels", "peas", "green peas",

    # Legumes
    "kidney bean", "black bean", "pinto bean", "navy bean",
    "lima bean", "cannellini bean", "great northern bean",
    "garbanzo bean", "chickpea", "chickpeas", "adzuki bean", "baked bean", "baked beans",
    "refried bean", "refried beans", "hummus", "lentil", "lentils",
    "split pea", "split peas", "black-eyed pea", "black-eyed peas",
    "fava bean", "fava beans", "edamame", "soy beans", "soybeans",

    # Sweeteners and sugars - ENHANCED
    "sugar", "brown sugar", "coconut sugar", "muscovado sugar",
    "demerara", "turbinado", "molasses", "honey", "agave nectar",
    "maple syrup", "agave syrup", "brown rice syrup",
    "corn syrup", "high fructose corn syrup", "light corn syrup", "dark corn syrup",
    "glucose syrup", "rice syrup", "barley malt syrup", "malt syrup",
    "golden syrup", "invert syrup", "cane syrup", "sorghum syrup",
    "date syrup", "coconut nectar", "yacon syrup",
    "powdered sugar", "confectioners sugar", "icing sugar",
    "raw sugar", "caster sugar", "superfine sugar", "rock sugar",

    # Sauces and condiments high in sugar
    "tomato sauce", "ketchup", "bbq sauce", "barbecue sauce", "teriyaki sauce",
    "hoisin sauce", "sweet chili sauce", "sweet pickle", "sweet and sour sauce",
    "sweet relish", "sweet soy glaze", "marmalade", "jam", "jelly", "preserves",
    "cranberry sauce", "apple sauce", "applesauce", "chutney",
    "glazed", "honey mustard", "sweet mustard",

    # Processed foods and snacks
    "bread", "white bread", "whole wheat bread", "bagel", "muffin", "cookie", "cookies", "cooky", "cake",
    "pastry", "pie crust", "pizza", "pizza crust", "pizza flour", "pizza dough",
    "naan", "pita", "pita bread", "roti", "chapati", "tortilla", "flour tortilla",
    "pretzel", "chip", "chips", "potato chip", "corn chip", "french fry", "french fries", "tater tot",
    "doughnut", "donut", "graham cracker", "crackers", "hamburger bun", "hot-dog bun", "bun",
    "pancake", "waffle", "crepe", "biscuit", "scone", "roll", "dinner roll",
    "cereal bar", "granola bar", "protein bar", "power bar",

    # Breakfast items
    "breakfast cereal", "cereal", "granola", "muesli", "energy bar",
    "oatmeal", "porridge", "pancake mix", "waffle mix",

    # Beverages
    "soy sauce", "orange juice", "apple juice", "grape juice", "cranberry juice",
    "fruit punch", "lemonade", "chocolate milk", "flavored milk",
    "sweetened condensed milk", "sweetened cranberry", "sweetened yogurt",
    "sports drink", "energy drink", "soda", "soft drink", "cola",
    "fruit smoothie", "milkshake", "frappuccino",

    # Alcoholic beverages (carbs from alcohol and mixers)
    "ale", "beer", "lager", "ipa", "pilsner", "stout", "porter", "wheat beer",
    "moscato", "riesling", "port", "sherry", "sangria", "dessert wine",
    "margarita", "mojito", "pina colada", "daiquiri", "mai tai",
    "cosmopolitan", "whiskey sour", "bloody mary", "long island iced tea",
    "bailey", "baileys", "kahlua", "amaretto", "frangelico", "limoncello",
    "triple sec", "curacao", "alcoholic lemonade", "alcopop", "wine cooler",
    "breezer", "smirnoff ice", "mike hard lemonade", "hard cider", "cider",
    "mead", "sake", "plum wine",

    # Specialty items
    "tapioca", "tapioca starch", "arrowroot", "job's tear", "jobs tear", "job tear",
    "gnocchi", "tempura batter", "breading", "breadcrumbs", "panko",
    "ice cream", "gelato", "sorbet", "sherbet", "frozen yogurt",
    "candy", "hard candy", "gummy bear", "gummy candy", "chocolate bar",
    "milk chocolate", "caramel", "toffee", "fudge", "nougat", "marshmallow",

    # Soy products (often sweetened)
    "soybean sweetened", "teriyaki tofu", "sweetened soy milk",

    # Additional high-carb items
    "cornstarch", "potato starch", "wheat starch", "modified food starch",
    "flour tortilla", "corn tortilla", "rice cake", "rice cakes", "rice noodle", "rice noodles",
    "ramen", "udon", "soba", "lo mein", "pad thai", "pho",
    "risotto", "pilaf", "stuffing", "dressing",
    "hashbrown", "hash brown", "mashed potato", "scalloped potato",
]))

# Animal-derived ingredients that disqualify vegan classification
NON_VEGAN = list(set([
    "kahlua", "coffee liqueur", "coffee-flavored liqueur",
    'ox', 'oxen',  # Cattle used as draft animals

    # Meat - Red meat
    'beef', 'steak', 'ribeye', 'sirloin', 'veal', 'lamb', 'mutton',
    'pork', 'bacon', 'ham', 'boar', 'goat', 'kid', 'venison',
    'rabbit', 'hare',

    # Meat - Poultry
    'chicken', 'turkey', 'duck', 'goose', 'quail', 'pheasant',
    'partridge', 'grouse',

    # Meat - Organ meats
    'liver', 'kidney', 'heart', 'tongue', 'brain', 'sweetbread',
    'tripe', 'gizzard', 'offal', 'bone', 'marrow', 'oxtail',

    # Meat - Processed (ENHANCED)
    'sausage', 'bratwurst', 'knackwurst', 'mettwurst',
    'salami', 'pepperoni', 'pastrami', 'bresaola',
    'prosciutto', 'pancetta', 'guanciale', 'speck',
    'mortadella', 'capocollo', 'coppa', 'cotechino',
    'chorizo', 'lard', 'tallow',

    # Hot dog variations (NEW)
    'hot dog', 'hotdog', 'hot dogs', 'hotdogs', 'frankfurter', 'frankfurters',
    'wiener', 'wieners', 'vienna sausage', 'cocktail sausage',

    # Additional processed meats (NEW)
    'bologna', 'liverwurst', 'head cheese', 'spam', 'corned beef',
    'beef jerky', 'turkey jerky', 'venison jerky', 'biltong',
    'pate', 'foie gras', 'terrine', 'rillettes', 'confit',
    'kielbasa', 'andouille', 'boudin', 'blood sausage', 'black pudding',
    'haggis', 'scrapple', 'chicharron', 'pork rind', 'pork rinds',
    'crackling', 'cracklings',

    # Fish and Seafood
    'fish', 'salmon', 'tuna', 'cod', 'haddock', 'halibut',
    'mackerel', 'herring', 'sardine', 'anchovy', 'trout',
    'tilapia', 'catfish', 'carp', 'sole', 'snapper', 'eel',
    'shrimp', 'prawn', 'crab', 'lobster', 'langoustine',
    'clam', 'mussel', 'oyster', 'scallop', 'squid', 'calamari',
    'octopus', 'krill', 'caviar', 'roe',
    'fishpaste', 'shrimppaste', 'anchovypaste', 'bonito',
    'katsuobushi', 'dashi', 'nampla',

    # Additional seafood (NEW)
    'crayfish', 'crawfish', 'abalone', 'conch', 'sea urchin', 'uni',
    'fish sauce', 'oyster sauce', 'fish stock', 'seafood stock',
    'surimi', 'crab stick', 'fish cake', 'fish ball',

    # Dairy - Milk products
    'milk', 'cream', 'butter', 'buttermilk', 'condensed', 'evaporated',
    'lactose', 'whey', 'casein', 'ghee', 'kefir',

    # Additional dairy (NEW)
    'half and half', 'heavy cream', 'whipping cream', 'clotted cream',
    'milk powder', 'milk solids', 'dairy', 'lactalbumin', 'lactoglobulin',
    'milk protein', 'milk fat', 'milkfat', 'butter fat', 'butterfat',

    # Dairy - Cheese
    'cheese', 'cheddar', 'mozzarella', 'parmesan', 'parmigiano',
    'reggiano', 'pecorino', 'ricotta', 'mascarpone',
    'brie', 'camembert', 'roquefort', 'gorgonzola', 'stilton',
    'emmental', 'gruyere', 'fontina', 'asiago', 'manchego',
    'halloumi', 'feta', 'quark', 'paneer', 'stracciatella',
    'provolone', 'taleggio',

    # Additional cheeses (NEW)
    'gouda', 'edam', 'jarlsberg', 'havarti', 'muenster', 'colby',
    'monterey jack', 'pepper jack', 'swiss cheese', 'american cheese',
    'velveeta', 'cheese whiz', 'nacho cheese', 'cotija', 'oaxaca',
    'boursin', 'chevre', 'fromage', 'burrata', 'comte', 'raclette',

    # Dairy - Other
    'yogurt', 'sourcream', 'cremefraiche', 'curd', 'custard',
    'icecream', 'gelatin', 'collagen',

    # Additional dairy products (NEW)
    'yoghurt', 'frozen yogurt', 'lassi', 'ayran', 'buttermilk',
    'sour cream', 'creme fraiche', 'clotted cream', 'double cream',
    'single cream', 'table cream', 'coffee cream', 'pudding',
    'panna cotta', 'creme brulee', 'mousse', 'tiramisu',

    # Eggs
    'egg', 'eggs', 'yolk', 'albumen', 'omelet', 'omelette', 'meringue',

    # Additional egg products (NEW)
    'egg white', 'egg whites', 'egg yolk', 'egg yolks', 'whole egg',
    'egg powder', 'dried egg', 'liquid egg', 'egg wash', 'beaten egg',
    'scrambled egg', 'fried egg', 'poached egg', 'deviled egg',
    'egg noodle', 'egg pasta', 'egg drop', 'egg salad', 'egg sandwich',
    'quiche', 'frittata', 'shakshuka', 'egg custard', 'egg nog', 'eggnog',
    'hollandaise', 'bearnaise', 'carbonara',

    # Other animal products (ENHANCED)
    'honey', 'shellfish', 'escargot', 'snail', 'frog',
    'worcestershire', 'aioli', 'mayonnaise',
    'broth', 'stock', 'gravy',

    # Additional animal products (NEW)
    'beeswax', 'royal jelly', 'propolis', 'lanolin', 'lard',
    'tallow', 'schmaltz', 'duck fat', 'goose fat', 'bone broth',
    'chicken broth', 'beef broth', 'fish broth', 'demi glace',
    'consomme', 'aspic', 'isinglass', 'carmine', 'cochineal',
    'shellac', 'vitamin d3', 'omega 3', 'fish oil', 'cod liver oil',
    'rennet', 'pepsin', 'lipase', 'animal enzyme', 'animal enzymes',
    'mono and diglycerides', 'monoglycerides', 'diglycerides',
    'stearic acid', 'lactic acid', 'glycerin', 'glycerine',
    'vitamin a palmitate', 'retinol', 'cholecalciferol',
]))


# Regex patterns for keto-friendly ingredients (overrides blacklist)
KETO_WHITELIST = [

    r"\bavocado\b",    # Singular form (to catch lemmatized version)
    r"\bavocados\b",   # Plural form
    r"\bgreen onion\b",
    r"\bgreen onions\b",
    r"\bscallion\b",   # Alternative name for green onion
    r"\bscallions\b",
    r"\bcumin\b",      # Spice - virtually no carbs
    r"\bonion\b",      # Base form
    r"\bonions\b",     # Plural form

    # Other common keto spices/herbs that might be missing:
    r"\bbasil\b",
    r"\boregano\b",
    r"\bthyme\b",
    r"\brosemary\b",
    r"\bsage\b",
    r"\bparsley\b",
    r"\bcilantro\b",
    r"\bcoriander\b",
    r"\bpaprika\b",
    r"\bturmeric\b",
    r"\bcinnamon\b",  # Small amounts
    r"\bnutmeg\b",
    r"\bcardamom\b",
    r"\bginger\b",  # Ground ginger
    r"\bgarlic powder\b",
    r"\bonion powder\b",
    r"\bchili powder\b",
    r"\bcayenne\b",
    r"\bblack pepper\b",
    r"\bwhite pepper\b",
    r"\bred pepper flakes\b",

    # Mustard varieties (very low carb)
    r"\bmustard\b",
    r"\bground mustard\b",
    r"\bmustard powder\b",
    r"\bmustard seed\b",
    r"\bmustard seeds\b",
    r"\bdijon mustard\b",
    r"\byellow mustard\b",
    r"\bbrown mustard\b",
    r"\bwhole grain mustard\b",
    r"\bspicy mustard\b",
    r"\bprepared mustard\b",
    r"\bdry mustard\b",

    # Common keto vegetables that might be missing:
    r"\basparagus\b",
    r"\bcelery\b",
    r"\bkale\b",
    r"\barugula\b",
    r"\blettuce\b",
    r"\bchard\b",
    r"\bcollard greens\b",
    r"\bradish\b",
    r"\bradishes\b",
    r"\bbok choy\b",
    r"\bcabbage\b",
    r"\bsauerkraut\b",
    r"\bpickle\b",
    r"\bpickles\b",  # If sugar-free
    r"\bdill pickle\b",
    r"\bbell pepper\b",
    r"\bbell peppers\b",
    r"\bjalapeno\b",
    r"\bjalapenos\b",
    r"\bserrano\b",
    r"\bhabanero\b",

    # Common keto fats/oils:
    r"\bghee\b",
    r"\blard\b",
    r"\btallow\b",
    r"\bavocado oil\b",
    r"\bmct oil\b",
    r"\bflaxseed oil\b",
    r"\bwalnut oil\b",
    r"\bsesame oil\b",

    # Keto proteins:
    r"\begg\b",
    r"\beggs\b",
    r"\bbeef\b",
    r"\bpork\b",
    r"\blamb\b",
    r"\bveal\b",
    r"\bturkey\b",
    r"\bduck\b",
    r"\bvenison\b",
    r"\bbison\b",
    r"\btuna\b",
    r"\bsardine\b",
    r"\bsardines\b",
    r"\banchovies\b",
    r"\banchovy\b",
    r"\bherring\b",
    r"\btrout\b",
    r"\bhalibut\b",
    r"\bcod\b",
    r"\btilapia\b",
    r"\bshrimp\b",
    r"\bprawns\b",
    r"\blobster\b",
    r"\bcrab\b",
    r"\bscallops\b",
    r"\bmussels\b",
    r"\bclams\b",
    r"\boysters\b",


    r"\bheavy cream\b",
    r"\bwhipping cream\b",
    r"\bdouble cream\b",

    r"\bchicken\b",
    r"\bgarlic\b",
    r"\bparmesan\b",
    r"\bpine nuts\b",
    r"\bbutter\b",
    r"\bricotta\b",
    r"\bmozzarella\b",
    r"\bolive oil\b",
    r"\bmayonnaise\b",

    r"\bsalt\b",
    r"\bpepper\b",
    r"\bblack pepper\b",
    r"\bsalt and pepper\b",
    r"\bground black pepper\b",
    r"\bblue cheese\b",
    r"\blight cream\b",

    r"\bwalnuts\b",
    r"\bgreen beans\b",
    r"\blemon peel\b",
    r"\blemon zest\b",
    r"\bsplenda\b",
    r"\bartificial sweetener\b",
    r"\bsugar substitute\b",
    r"\bsherry\b",  # Small cooking amounts



    # Keto-friendly flours
    r"\balmond flour\b",
    r"\bcoconut flour\b",
    r"\bflaxseed flour\b",
    r"\bchia flour\b",
    r"\bsunflower seed flour\b",
    r"\bpeanut flour\b",
    r"\bhemp flour\b",
    r"\bsesame flour\b",
    r"\bwalnut flour\b",
    r"\bpecan flour\b",
    r"\bmacadamia flour\b",
    r"\bhazelnut flour\b",

    # Low-carb citrus exceptions
    r"\blemon juice\b",

    # Keto milk alternatives
    r"\balmond milk\b",
    r"\bcoconut milk\b",
    r"\bflax milk\b",
    r"\bmacadamia milk\b",
    r"\bhemp milk\b",
    r"\bcashew milk\b",
    r"\balmond cream\b",
    r"\bcoconut cream\b",
    r"\bsour cream\b",

    # Nut and seed butters
    r"\balmond butter\b",
    r"\bpeanut butter\b",
    r"\bcoconut butter\b",
    r"\bmacadamia butter\b",
    r"\bpecan butter\b",
    r"\bwalnut butter\b",
    r"\bhemp butter\b",

    # Keto bread alternatives
    r"\balmond bread\b",
    r"\bcoconut bread\b",
    r"\bcloud bread\b",
    r"\bketo bread\b",

    # Sugar-free sweeteners
    r"\bcoconut sugar[- ]free\b",
    r"\bstevia\b",
    r"\berytritol\b",
    r"\bmonk fruit\b",
    r"\bswerve\b",
    r"\ballulose\b",
    r"\bxylitol\b",
    r"\bsugar[- ]free\b",

    # Low-carb alternatives
    r"\bcauliflower rice\b",
    r"\bshirataki noodles\b",
    r"\bzucchini noodles\b",
    r"\bkelp noodles\b",
    r"\bsugar[- ]free chocolate\b",
    r"\bketo chocolate\b",

    # Low-carb vegetables and foods
    r"\bcacao\b",
    r"\bcocoa powder\b",
    r"\bketo ice[- ]cream\b",
    r"\bsugar[- ]free ice[- ]cream\b",
    r"\bjicama\b",
    r"\bzucchini\b",
    r"\bcucumber\b",
    r"\bbroccoli\b",
    r"\bcauliflower\b",
]

# Regex patterns for vegan-friendly ingredients (overrides blacklist)
VEGAN_WHITELIST = [
    r"\bkidney beans\b",

    # Vegan indicators (NEW)
    r"\bmeatless\b",
    r"\bmeat[- ]?free\b",
    r"\bplant[- ]?based\b",
    r"\bvegan\b",
    r"\bvegetarian\b",
    r"\bnon[- ]?dairy\b",
    r"\bdairy[- ]?free\b",
    r"\begg[- ]?free\b",
    r"\banimal[- ]?free\b",
    r"\bcruelty[- ]?free\b",

    # Meat alternatives (NEW)
    r"\bfaux\s+(?:meat|chicken|beef|pork|turkey|bacon|sausage)\b",
    r"\bfake\s+(?:meat|chicken|beef|pork|turkey|bacon|sausage)\b",
    r"\bmock\s+(?:meat|chicken|beef|pork|turkey|duck)\b",
    r"\bimitation\s+(?:meat|chicken|beef|pork|crab|shrimp)\b",
    r"\bvegetarian\s+(?:meat|chicken|beef|pork|bacon|sausage)\b",
    r"\bsoy\s+(?:meat|chicken|beef|pork|bacon|sausage|chorizo)\b",
    r"\bwheat\s+(?:meat|protein)\b",
    r"\bseitan\b",
    r"\btempeh\b",
    r"\btofu\b",
    r"\btextured\s+vegetable\s+protein\b",
    r"\btvp\b",
    r"\btsp\b",  # textured soy protein
    r"\bquorn\b",
    r"\bgardein\b",
    r"\bbeyond\s+(?:meat|burger|sausage)\b",
    r"\bimpossible\s+(?:meat|burger|sausage)\b",
    r"\btofurky\b",
    r"\btofurkey\b",
    r"\bfield\s+roast\b",
    r"\blightlife\b",
    r"\bmorningstar\b",
    r"\bboca\b",
    r"\bsimple\s+truth\s+plant[- ]?based\b",

    # Egg exceptions (plant-based)
    r"\beggplant\b",
    r"\begg\s*fruit\b",
    r"\bvegan\s+egg\b",
    r"\begg\s+replacer\b",
    r"\bener[- ]?g\s+egg\s+replacer\b",
    r"\baqua[- ]?faba\b",
    r"\bflax\s+egg\b",
    r"\bchia\s+egg\b",
    r"\bjust\s+egg\b",

    # Milk exceptions (plant-based)
    r"\bmillet\b",  # grain, not milk
    r"\bmilk\s+thistle\b",
    r"\bcoconut\s+milk\b",
    r"\boat\s+milk\b",
    r"\bsoy\s+milk\b",
    r"\balmond\s+milk\b",
    r"\bcashew\s+milk\b",
    r"\brice\s+milk\b",
    r"\bhazelnut\s+milk\b",
    r"\bpea\s+milk\b",
    r"\bhemp\s+milk\b",
    r"\bflax\s+milk\b",
    r"\bmacadamia\s+milk\b",
    r"\bwalnut\s+milk\b",
    r"\bsunflower\s+milk\b",
    r"\bpumpkin\s+seed\s+milk\b",
    r"\bquinoa\s+milk\b",
    r"\bsesame\s+milk\b",
    r"\bpistachio\s+milk\b",
    r"\bbarley\s+milk\b",
    r"\bspelt\s+milk\b",
    r"\btiger\s+nut\s+milk\b",
    r"\bplant\s+milk\b",
    r"\bnon[- ]?dairy\s+milk\b",
    r"\bmylk\b",  # Common spelling for plant milks

    # Rice alternatives (vegetable-based)
    r"\bcauliflower rice\b",
    r"\bbroccoli rice\b",
    r"\bsweet potato rice\b",
    r"\bzucchini rice\b",
    r"\bcabbage rice\b",
    r"\bkonjac rice\b",
    r"\bshirataki rice\b",
    r"\bmiracle rice\b",
    r"\bpalmini rice\b",

    # Butter exceptions (plant-based)
    r"\bbutternut\b",  # squash
    r"\bbutterfly\s+pea\b",
    r"\bcocoa\s+butter\b",
    r"\bpeanut\s+butter\b",
    r"\balmond\s+butter\b",
    r"\bsunflower(?:\s*seed)?\s+butter\b",
    r"\bpistachio\s+butter\b",
    r"\bvegan\s+butter\b",
    r"\bplant\s+butter\b",
    r"\bnut\s+butter\b",
    r"\bseed\s+butter\b",
    r"\btahini\b",
    r"\bcashew\s+butter\b",
    r"\bhazelnut\s+butter\b",
    r"\bwalnut\s+butter\b",
    r"\bpecan\s+butter\b",
    r"\bmacadamia\s+butter\b",
    r"\bsoy\s+butter\b",
    r"\bearth\s+balance\b",
    r"\bmiyoko\b",
    r"\bcountry\s+crock\s+plant\b",

    # Honey exceptions (plants)
    r"\bhoneydew\b",
    r"\bhoneysuckle\b",
    r"\bhoneycrisp\b",
    r"\bhoney\s+locust\b",
    r"\bhoneyberry\b",
    r"\bagave\b",
    r"\bmaple\s+syrup\b",
    r"\bdate\s+syrup\b",
    r"\bbrown\s+rice\s+syrup\b",
    r"\bcoconut\s+nectar\b",

    # Cream exceptions (plant-based)
    r"\bcream\s+of\s+tartar\b",
    r"\bice[- ]cream\s+bean\b",
    r"\bcoconut\s+cream\b",
    r"\bcashew\s+cream\b",
    r"\bvegan\s+cream\b",
    r"\bplant\s+cream\b",
    r"\boat\s+cream\b",
    r"\bsoy\s+cream\b",
    r"\balmond\s+cream\b",
    r"\brice\s+cream\b",
    r"\bnon[- ]?dairy\s+cream\b",
    r"\bwhipped\s+coconut\b",
    r"\bcoconut\s+whip\b",

    # Cheese exceptions (plant-based)
    r"\bcheesewood\b",
    r"\bvegan\s+cheese\b",
    r"\bcashew\s+cheese\b",
    r"\bnut\s+cheese\b",
    r"\bplant\s+cheese\b",
    r"\bnon[- ]?dairy\s+cheese\b",
    r"\balmond\s+cheese\b",
    r"\bcoconut\s+cheese\b",
    r"\bsoy\s+cheese\b",
    r"\bnutritional\s+yeast\b",
    r"\bnooch\b",
    r"\bdaiya\b",
    r"\bviolife\b",
    r"\bchao\b",
    r"\bkite\s+hill\b",
    r"\bmiyoko\s+creamery\b",
    r"\bfollow\s+your\s+heart\b",

    # Fish exceptions (plants)
    r"\bfish\s+mint\b",
    r"\bfish\s+pepper\b",
    r"\bsilverfish\s+melon\b",
    r"\bfish\s+grass\b",

    # Beef exceptions (plants/mushrooms)
    r"\bbeefsteak\s+plant\b",
    r"\bbeefsteak\s+mushroom\b",
    r"\bbeefsteak\s+tomato\b",

    # Chicken/hen exceptions (mushrooms)
    r"\bchicken[- ]of[- ]the[- ]woods\b",
    r"\bchicken\s+mushroom\b",
    r"\bhen[- ]of[- ]the[- ]woods\b",

    # Meat exceptions (plants)
    r"\bsweetmeat\s+(?:pumpkin|squash)\b",
    r"\bmeatball\s+plant\b",

    # Bacon alternatives
    r"\bcoconut\s+bacon\b",
    r"\bmushroom\s+bacon\b",
    r"\bsoy\s+bacon\b",
    r"\bvegan\s+bacon\b",
    r"\brice\s+paper\s+bacon\b",
    r"\beggplant\s+bacon\b",
    r"\bcarrot\s+bacon\b",
    r"\btempeh\s+bacon\b",
    r"\bseitan\s+bacon\b",
    r"\btofu\s+bacon\b",

    # Mayo alternatives (NEW)
    r"\bvegan\s+mayo\b",
    r"\bvegenaise\b",
    r"\bplant\s+mayo\b",
    r"\begg[- ]?free\s+mayo\b",
    r"\baquafaba\s+mayo\b",
    r"\bsoy\s+mayo\b",
    r"\bjust\s+mayo\b",
    r"\bhellmann\s+vegan\b",

    # Stock/broth alternatives (NEW)
    r"\bvegetable\s+(?:stock|broth)\b",
    r"\bmushroom\s+(?:stock|broth)\b",
    r"\bmiso\s+(?:stock|broth|soup)\b",
    r"\bkombu\s+(?:stock|broth|dashi)\b",
    r"\bvegan\s+(?:stock|broth)\b",
    r"\bplant\s+(?:stock|broth)\b",
    r"\bno[- ]?chicken\s+(?:stock|broth|base)\b",
    r"\bno[- ]?beef\s+(?:stock|broth|base)\b",
]