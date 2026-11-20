"""Stopword lists used when building the multilingual song vocabulary."""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# remove spanish stop words
SPANISH_STOP_WORDS = {
    "al", "algo", "alguien", "alguna", "algún",
    "aquel", "aquella", "aquello", "aquí", "aqui",
    "así", "asi", "aun", "aún",
    "cada", "como", "cómo", "con", "contra",
    "cualquier", "cuando", "cuanto", "cuatro",
    "de", "del", "dentro", "dos", "dónde",
    "el", "ella", "ello",
    "en", "esa", "ese", "eso",
    "esta", "está", "están", "esté", "esto",
    "fue", "fuera", "fui",
    "gran",
    "ha", "han", "hasta", "hay", "he",
    "la", "las", "le", "les", "lo", "los",
    "mas", "más", "me", "mi", "mí", "mis",
    "misma", "mismo", "mucha", "mucho", "muy",
    "ni", "nos", "nuestra", "nuestro",
    "nunca",
    "o", "os", "otra", "otro",
    "para", "pero", "poco", "por",
    "que", "qué", "quien", "quién",
    "quienes", "quienes", "quiera",
    "se", "ser", "si", "sí", "sin",
    "sobre", "solo", "sólo", "su", "sus",
    "tal", "tambien", "también", "tan",
    "tanto", "te", "ti", "toda", "todo",
    "tras", "tres", "tu", "tú", "tus",
    "un", "una", "uno",
    "y", "ya", "yo", "él"
}

GERMAN_STOP_WORDS = {
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am",
    "an", "ander", "andere", "anderem", "anderen", "anderer", "anderes", "anderm",
    "andern", "anderr", "anders", "auch", "auf", "aus", "bei", "bin", "bis",
    "bist", "da", "damit", "dann", "das", "dass", "dasselbe", "dazu", "dein",
    "dem", "den", "denn", "der", "des", "die", "dies", "diese", "diesem",
    "diesen", "dieser", "dieses", "doch", "dort", "du", "durch", "ein", "eine",
    "einem", "einen", "einer", "eines", "einige", "einigen", "einiger", "einiges",
    "er", "es", "etwas", "für", "gegen", "gewesen", "hat", "habe", "haben",
    "hier", "hin", "hinter", "ich", "im", "in", "ist", "ja", "jede", "jedem",
    "jeden", "jeder", "jedes", "jene", "jenem", "jenen", "jener", "jenes",
    "jetzt", "kann", "kein", "keine", "mit", "muss", "nach", "nicht", "nichts",
    "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein", "seine", "selbst",
    "sich", "sie", "sind", "so", "solche", "solchem", "solchen", "solcher",
    "solches", "über", "um", "und", "uns", "unter", "vom", "von", "vor", "war",
    "waren", "was", "weg", "weil", "weiter", "welche", "welchem", "welchen",
    "welcher", "welches", "wenn", "wer", "werde", "werden", "wie", "wieder",
    "will", "wir", "wo", "zu", "zum", "zur",
}

FRENCH_STOP_WORDS = {
    "a", "ai", "aie", "aient", "aies", "ait", "as", "au", "aucun", "aucune",
    "aux", "avaient", "avais", "avait", "avec", "avez", "avons", "c", "ce",
    "ceci", "celà", "cela", "celle", "celles", "celui", "ces", "cet", "cette",
    "d", "dans", "de", "des", "du", "elle", "elles", "en", "es", "est", "et",
    "étaient", "étais", "était", "étant", "été", "êtes", "eu", "eue", "eues",
    "eurent", "eus", "eusse", "eussent", "eusses", "eut", "eux", "fut", "furent",
    "fus", "fusse", "fussent", "fusses", "fût", "ici", "il", "ils", "j", "je",
    "l", "la", "le", "les", "leur", "leurs", "lui", "m", "ma", "mais", "me",
    "mes", "moi", "mon", "même", "ne", "nos", "notre", "nous", "on", "ont", "ou",
    "par", "pas", "pour", "qu", "que", "quel", "quelle", "quelles", "quels",
    "qui", "s", "sa", "sans", "se", "sera", "serai", "seraient", "serais",
    "serait", "seras", "serez", "seriez", "serons", "soi", "soient", "sois",
    "soit", "sommes", "sont", "sur", "t", "ta", "te", "tes", "toi", "ton",
    "tous", "tout", "toute", "toutes", "tu", "un", "une", "vos", "votre", "vous",
    "y", "à",
}

PORTUGUESE_STOP_WORDS = {
    "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "até", "com", "como", "contra", "da", "das", "de", "dela", "delas",
    "dele", "deles", "depois", "do", "dos", "e", "é", "ela", "elas", "ele", "eles",
    "em", "entre", "era", "eram", "essa", "essas", "esse", "esses", "esta",
    "está", "estão", "estas", "este", "estes", "eu", "foi", "for", "foram", "há",
    "isso", "isto", "já", "la", "lá", "lhe", "lhes", "mais", "mas", "me", "mesmo",
    "meu", "meus", "minha", "minhas", "muito", "muitos", "na", "não", "nas", "nem",
    "nesse", "nesta", "neste", "no", "nos", "nós", "nossa", "nossas", "nosso",
    "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo",
    "pelos", "per", "pode", "por", "qual", "quando", "que", "quem", "se", "seja",
    "sem", "ser", "seu", "seus", "só", "sob", "sobre", "sua", "suas", "também",
    "te", "tem", "tendo", "tenho", "ter", "tua", "tuas", "um", "uma", "umas",
    "uns", "vai", "vamos", "vão", "você", "vocês",
}

STOP_WORDS_BY_LANGUAGE = {
    "german": GERMAN_STOP_WORDS,
    "french": FRENCH_STOP_WORDS,
    "portuguese": PORTUGUESE_STOP_WORDS,
    "spanish": SPANISH_STOP_WORDS,
}

ALL_STOP_WORDS = (
    set(ENGLISH_STOP_WORDS)
    | set().union(*STOP_WORDS_BY_LANGUAGE.values())
)
