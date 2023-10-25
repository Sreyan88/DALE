import pickle

def get_label2desc(dataset_name):
    if 'ng' in dataset_name:
        label2desc = {0:"alt atheism",
                1:"computer graphics",
                2:"computer os microsoft windows misc",
                3:"computer system ibm pc hardware",
                4:"computer system mac hardware",
                5:"computer windows x",
                6:"misc for sale",
                7:"rec autos auto",
                8:"rec motorcycles",
                9:"rec sport baseball",
                10:"rec sport hockey",
                11:"sci crypt",
                12:"sci electronics",
                13:"sci medicine med",
                14:"sci space universe",
                15:"soc religion christian",
                16:"talk politics guns gun",
                17:"talk politics mideast",
                18:"talk politics misc",
                19:"talk religion misc"}
    elif 'bbc' in dataset_name or '5huffpost' in dataset_name:
        label2desc = None
    elif 'yahoo' in dataset_name:
        label2desc = {0: "Society Culture",
                    1: "Science Mathematics",
                    2: "Health",
                    3: "Education Reference",
                    4: "Computers Internet",
                    5: "Sports",
                    6: "Business Finance",
                    7: "Entertainment Music",
                    8: "Family Relationships",
                    9: "Politics Government"}
    elif 'imdb' in dataset_name or 'sst2' in dataset_name:  
        label2desc = {0: "negative, bad", 1: "positive, good"}
    elif "ledgar" in dataset_name:
        label2desc = {0: 'Adjustments', 1: 'Agreements', 2: 'Amendments', 3: 'Anti-Corruption Laws', 4: 'Applicable Laws', 5: 'Approvals', 6: 'Arbitration', 7: 'Assignments', 8: 'Assigns', 9: 'Authority', 10: 'Authorizations', 11: 'Base Salary', 12: 'Benefits', 13: 'Binding Effects', 14: 'Books', 15: 'Brokers', 16: 'Capitalization', 17: 'Change In Control', 18: 'Closings', 19: 'Compliance With Laws', 20: 'Confidentiality', 21: 'Consent To Jurisdiction', 22: 'Consents', 23: 'Construction', 24: 'Cooperation', 25: 'Costs', 26: 'Counterparts', 27: 'Death', 28: 'Defined Terms', 29: 'Definitions', 30: 'Disability', 31: 'Disclosures', 32: 'Duties', 33: 'Effective Dates', 34: 'Effectiveness', 35: 'Employment', 36: 'Enforceability', 37: 'Enforcements', 38: 'Entire Agreements', 39: 'Erisa', 40: 'Existence', 41: 'Expenses', 42: 'Fees', 43: 'Financial Statements', 44: 'Forfeitures', 45: 'Further Assurances', 46: 'General', 47: 'Governing Laws', 48: 'Headings', 49: 'Indemnifications', 50: 'Indemnity', 51: 'Insurances', 52: 'Integration', 53: 'Intellectual Property', 54: 'Interests', 55: 'Interpretations', 56: 'Jurisdictions', 57: 'Liens', 58: 'Litigations', 59: 'Miscellaneous', 60: 'Modifications', 61: 'No Conflicts', 62: 'No Defaults', 63: 'No Waivers', 64: 'Non-Disparagement', 65: 'Notices', 66: 'Organizations', 67: 'Participations', 68: 'Payments', 69: 'Positions', 70: 'Powers', 71: 'Publicity', 72: 'Qualifications', 73: 'Records', 74: 'Releases', 75: 'Remedies', 76: 'Representations', 77: 'Sales', 78: 'Sanctions', 79: 'Severability', 80: 'Solvency', 81: 'Specific Performance', 82: 'Submission To Jurisdiction', 83: 'Subsidiaries', 84: 'Successors', 85: 'Survival', 86: 'Tax Withholdings', 87: 'Taxes', 88: 'Terminations', 89: 'Terms', 90: 'Titles', 91: 'Transactions With Affiliates', 92: 'Use Of Proceeds', 93: 'Vacations', 94: 'Venues', 95: 'Vesting', 96: 'Waiver Of Jury Trials', 97: 'Waivers', 98: 'Warranties', 99: 'Withholdings'}
    elif "scotus" in dataset_name:
        label2desc = {0: 'Criminal Procedure', 1: 'Civil Rights', 2: 'First Amendment', 3: 'Due Process', 4: 'Privacy', 5: 'Attorneys', 6: 'Unions', 7: 'Economic Activity', 8: 'Judicial Power', 9: 'Federalism', 10: 'Interstate Relations', 11: 'Federal Taxation', 12: 'Miscellaneous', 13: 'Private Action'}
    elif "eurlex" in dataset_name:
        label2desc = {'100163': 'political framework', '100168': 'politics and public safety', '100169': 'executive power and public service', '100170': 'international affairs', '100171': 'cooperation policy', '100172': 'international security', '100173': 'defence', '100174': 'EU institutions and European civil service', '100175': 'European Union law', '100176': 'European construction', '100177': 'EU finance', '100179': 'civil law', '100180': 'criminal law', '100183': 'international law', '100184': 'rights and freedoms', '100185': 'economic policy', '100186': 'economic conditions', '100187': 'regions and regional policy', '100189': 'national accounts', '100190': 'economic analysis', '100191': 'trade policy', '100192': 'tariff policy', '100193': 'trade', '100194': 'international trade', '100195': 'consumption', '100196': 'marketing', '100197': 'distributive trades', '100198': 'monetary relations', '100199': 'monetary economics', '100200': 'financial institutions and credit', '100201': 'free movement of capital', '100202': 'financing and investment', '100204': 'public finance and budget policy', '100205': 'budget', '100206': 'taxation', '100207': 'prices', '100212': 'social affairs', '100214': 'social protection', '100215': 'health', '100220': 'documentation', '100221': 'communications', '100222': 'information and information processing', '100223': 'information technology and data processing', '100224': 'natural and applied sciences', '100226': 'business organisation', '100227': 'business classification', '100229': 'management', '100230': 'accounting', '100231': 'competition', '100232': 'employment', '100233': 'labour market', '100234': 'organisation of work and working conditions', '100235': 'personnel management and staff remuneration', '100237': 'transport policy', '100238': 'organisation of transport', '100239': 'land transport', '100240': 'maritime and inland waterway transport', '100241': 'air and space transport', '100242': 'environmental policy', '100243': 'natural environment', '100244': 'deterioration of the environment', '100245': 'agricultural policy', '100246': 'agricultural structures and production', '100247': 'farming systems', '100248': 'cultivation of agricultural land', '100249': 'means of agricultural production', '100250': 'agricultural activity', '100252': 'fisheries', '100253': 'plant product', '100254': 'animal product', '100255': 'processed agricultural produce', '100256': 'beverages and sugar', '100257': 'foodstuff', '100258': 'agri-foodstuffs', '100259': 'food technology', '100260': 'production', '100261': 'technology and technical regulations', '100262': 'research and intellectual property', '100263': 'energy policy', '100264': 'coal and mining industries', '100265': 'oil industry', '100266': 'electrical and nuclear industries', '100268': 'industrial structures and policy', '100269': 'chemistry', '100270': 'iron steel and other metal industries', '100271': 'mechanical engineering', '100272': 'electronics and electrical engineering', '100273': 'building and public works', '100274': 'wood industry', '100275': 'leather and textile industries', '100276': 'miscellaneous industries', '100277': 'Europe', '100278': 'regions of EU Member States', '100279': 'America', '100280': 'Africa', '100281': 'Asia and Oceania', '100282': 'economic geography', '100283': 'political geography', '100284': 'overseas countries and territories', '100285': 'United Nations'}
    elif "unfair_tos" in dataset_name:
        label2desc = {0: 'Limitation of liability', 1: 'Unilateral termination', 2: 'Unilateral change', 3: 'Content removal', 4: 'Contract by using', 5: 'Choice of law', 6: 'Jurisdiction', 7: 'Arbitration', 8: 'Fair'}
    elif "ecthr" in dataset_name:
        label2desc = {
            2: "Right to life",
            3: "Prohibition of torture",
            5: "Right to liberty and security",
            6: "Right to a fair trial",
            8: "Right to respect for private and family life",
            9: "Freedom of thought conscience and religion",
            10: "Freedom of expression",
            11: "Freedom of assembly and association",
            14: "Prohibition of discrimination",
            "P1-1": "Protection of property",
            "NO": "No violations"
        }
    elif "ildc" in dataset_name:
        label2desc = {0 : "all petitions have been rejected", 1: "atleast one petition has been accepted"}
    elif "20NG" in dataset_name:
        label2desc = {
            'comp.sys.mac.hardware': 'Computer Systems Mac Hardware',
            'talk.politics.mideast': 'Talk Politics Middle East',
            'sci.electronics': 'Science Electronics',
            'rec.sport.baseball': 'Recreation Sports Baseball',
            'talk.politics.misc': 'Talk Politics Miscellaneous',
            'rec.sport.hockey': 'Recreation Sports Hockey',
            'sci.space': 'Science Space',
            'comp.os.ms-windows.misc': 'Computer Systems MS Windows Miscellaneous',
            'comp.sys.ibm.pc.hardware': 'Computer Systems IBM PC Hardware',
            'sci.crypt': 'Science Cryptography',
            'misc.forsale': 'Miscellaneous For Sale',
            'comp.windows.x': 'Computer Systems Windows X',
            'talk.religion.misc': 'Talk Religion Miscellaneous',
            'sci.med': 'Science Medicine',
            'alt.atheism': 'Alternative Atheism',
            'comp.graphics': 'Computer Systems Graphics',
            'soc.religion.christian': 'Society Religion Christian',
            'rec.motorcycles': 'Recreation Motorcycles',
            'talk.politics.guns': 'Talk Politics Guns',
            'rec.autos': 'Recreation Autos'
        }
    elif "ots_topics" in dataset_name:
        label2desc = {1:"arbitration", 2:"unilateral change", 3:"content removal", 4:"jurisdiction", 
                    5:"choice of law", 6:"limitation of liability", 7:"unilateral termination",
                    8:"contract by using", 9:"privacy included", 10: "Fair"}
    elif "ots" in dataset_name:
        label2desc = {0: "potentially unfair", 1: "clearly unfair", 2: "clearly fair"}
    elif "rr" in dataset_name:
        label2desc = {'PREAMBLE': 'PREAMBLE', 'NONE': 'NONE', 'FAC': 'FAC', 'ARG_RESPONDENT': 'ARG_RESPONDENT', 'RLC': 'RLC', 'ARG_PETITIONER': 'ARG_PETITIONER', 'ANALYSIS': 'ANALYSIS', 'PRE_RELIED': 'PRE_RELIED', 'RATIO': 'RATIO', 'RPC': 'RPC', 'ISSUE': 'ISSUE', 'STA': 'STA', 'PRE_NOT_RELIED': 'PRE_NOT_RELIED'}
    else:
        print(f"{dataset_name} not supported! Please add the label info into `baselines/label_desc.py`")
    return label2desc