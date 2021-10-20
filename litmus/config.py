"""
	Tool supported config
"""
SUPPORTED_MODELS = ['mbert', 'xlmr']
SUPPORTED_TRAIN_ALGOS_V2 = ['xgboost']
SUPPORTED_MODES = ["interactive", "heatmap", "suggestions"]
SUPPORTED_DATA_FORMATS = ["matrix", "decomposed"]

"""
	Supported langs / model
"""
mbert_language_iso_codes = ['af', 'sq', 'ar', 'hy', 'ast', 'az', 'ba', 'eu', 'bar', 'be', 'bn', 'bs', 'br', 'bg', 'my', 'ca', 'ceb', 'ce', 'zh', 'cv', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'he', 'hi', 'hu', 'is', 'id', 'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'ky', 'ko', 'lv', 'lt', 'nds', 'lb', 'mk', 'mg', 'ms', 'ml', 'mr', 'min', 'ne', 'new', 'no', 'oc', 'fa', 'pms', 'pl', 'pt', 'pa', 'ro', 'ru', 'sk', 'sl', 'es', 'su', 'sw', 'sv', 'tl', 'tg', 'ta', 'tt', 'te', 'tr', 'uk', 'ur', 'uz', 'vi', 'war', 'cy', 'fy', 'yo', 'th']
xlmr_language_codes = ['af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'om', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'su', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'zh']
MODEL2LANGS = {'mbert': mbert_language_iso_codes, 'xlmr': xlmr_language_codes}

"""
	Families grouped by languages
	(used for ordering langs in heatmap)
"""
LANG_FAMILIES = [

	# Afro-Asiatic
		## Semitic
		["ar","he","am","mt"],
		## Cushitic and Chadic
		["om","so","ha"],

	# Tai-Kadak
	["th"],

	# Austroasiatic
	["vi","km"],

	# Turkic
		## Oghuz
		["tr","az"],
		## Kipchak
		["tt","ba","ky","kk"],
		## Karluk
		["uz","ug"],
		## Oghur
		["cv"],

	# Austronesian
		## Philippine
		["tl","ceb","war"],
		## Remaining Malayo-Polynesian
		["ms","id","jv","min","su","mg"],

	# Basque
	["eu"],

	# Uralic
	["et","fi","hu"],

	# Niger-Congo
	["yo","sw","xh"],

	# Indo European
		## Germanic
			### West Germanic
				#### Dutch
				["af","nl"],
				#### English
				["en","fy","nds"],
				#### German
				["de","yi","lb","bar"],
			### North Germanic
			["da","sv","no","is"],
		## Italic
			### Romance
				#### Iberian
				["pt","gl","ast","es"],
				#### Occitittan
				["ca","oc"],
				#### Italic
				["it","lo","pms"],
			### Misc
			["fr","ro"],
		## Celtic
		["ga","gd","cy","br"],
		## Balto-Slavic
			### Baltic
			["lv","lt"],
			### Slavic
				#### West
				["pl","cs","sk"],
				#### South
				["sl","bs","bg","mk"],
				#### East
				["be","ru","uk"],
		## Misc
		["el","sq","hy"],
		## Iranian
		["ckb","ku","fa","ps","tg"],
		## Indo-Aryan
			### North
			["ne"],
			### NWest/West
			["pa","sd","gu"],
			### Central
			["hi","ur"],
			### East
			["or","bn","as"],
			### South
			["mr","si"],

	# Dravidian
	["te","kn","ta","ml"],

	# Sino-Tibetian
	["zh","my","new"],

	# Japonic
	["ja"],

	# Koreanic
	["ko"],

	# Creole
	["ht"],

	# Northeast-Caucasian
	["ce"],

	# Kartvelian
	["ka"],
]
FAMILIES_FLATTENED = [lang for group in LANG_FAMILIES for lang in group]
LANG2INDEX = { lang: idx for idx,lang in enumerate(FAMILIES_FLATTENED) }