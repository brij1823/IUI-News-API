from flask import Flask, jsonify, request

from functools import lru_cache
import re

import spacy
from newspaper import Article
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from textblob import TextBlob

import codecs
from nltk import ne_chunk, pos_tag, word_tokenize

app = Flask(__name__)

'''
Assigns entities in a news article to roles of hero, villain, or victim.
'''


from stop_words import STOP_WORDS
from similarity_dictionary import SIM_DIC

'''
Recognizes the entities in a news article with a headline and determines
which of these entities are the  most relevant.
'''


RECOGNIZED_TYPES = ["PERSON", "ORGANIZATION", "GPE", "POSITION"]

NAME_PREFIXES = (
    'mr',
    'mrs',
    'ms',
    'miss',
    'dr',
    'doctor',
    'sgt',
    'sergeant',
    'rev',
    'reverend',
    'chief',
    'executive',
    'officer',
    'president',
)


class Entity:
    '''
    Represents an entity in the news article. Contain the entities top-level
    name, all name forms used in the article, count of occurences, and data
    about its locations within the article and headline.
    '''
    name = ""
    normalized_name = ""
    count = 0
    locations = {}
    headline_locations = []
    headline = False
    name_forms = []
    role = None
    relevance_score = 0

    def __init__(self, name, normalized_name, sentence_number=None,
                 index_list=None, headline=False, headline_index_list=None,
                 ):
        '''
        Construct a new entity.
        '''
        self.name = name
        self.normalized_name = normalized_name
        self.count = 1
        if sentence_number is not None and index_list is not None:
            self.locations = {sentence_number: index_list}
        if headline:
            self.headline = True
        if headline_index_list is not None:
            self.headline_locations = headline_index_list
        self.name_forms = [name]

    def __repr__(self):
        '''
        String representation of the entity.
        '''
        return ('(Name: {name}, Count: {count}, Headline: {headline}, '
                'Headline Locations: {headline_locs}, Text Locations: {locations}), '
                'Relevance Score: {relevance}'
                .format(name=self.name, count=self.count, headline=self.headline,
                        locations=self.locations, headline_locs=self.headline_locations,
                        relevance=round(self.relevance_score, 3),
                        )
                )


def get_locations(name, tokens, locations_found):
    '''
    Returns a list of indeces of the locations of name in the given tokenized
    set of words. Location are represented either as an index of the location
    (if the occurence is one word) or a tule of the first and last index of the
    entity (if the occurence is multiple words). Updates locations_found
    dictionary accordingly.
    '''
    index_list = []
    lastIndex = locations_found.get(name, 0)
    length = len(name.split())
    for j in range(lastIndex, len(tokens) - length + 1):
        if " ".join(tokens[j:j + length]) == name:
            locations_found[name] = j + length
            if length == 1:
                index_list.append(j)
            else:
                index_list.append((j, j + length - 1))
    return index_list


def extract_entities_article(tokenized_article):
    '''
    Returns a tuple where the first item is a list of (unmerged) entities from
    the article and the second item is the number of sentences in the article.

    Each entity is a tuple (entity name, sentence number, locations in sentence)
    '''
    named_entities = []
    num_sentences = len(tokenized_article)
    for sentence_number in range(num_sentences):
        sentence = tokenized_article[sentence_number]
        tokens = word_tokenize(sentence)
        tagged_sentence = pos_tag(tokens)
        chunked_entities = ne_chunk(tagged_sentence)

        locations_found = {}
        for tree in chunked_entities:
            if hasattr(tree, 'label') and tree.label() in RECOGNIZED_TYPES:
                entity_name = ' '.join(c[0] for c in tree.leaves())
                index_list = get_locations(entity_name, tokens, locations_found)
                entity = (entity_name, sentence_number, index_list)
                named_entities.append(entity)

    return (named_entities, num_sentences)


def merge_entities(temp_entities):
    '''
    Merges the list of temporary entity tuples into a list of Entity objects.
    Basis of merging algorithm from from NU Infolab News Context Project.
    (https://github.com/NUinfolab/context).
    '''
    merged_entities = []
    for temp_entity in temp_entities:
        name, sentence_number, index_list = temp_entity
        normalized_name = normalize_name(name)
        matches = []
        for entity in merged_entities:
            # Immediate match if name matches previously used name form
            if normalized_name == entity.normalized_name or normalized_name in entity.name_forms:
                matches = [entity]
                break
            # Get entities of which name is substring
            if normalized_name in entity.normalized_name:
                matches.append(entity)

        # If name matches one existing entity, merge it
        if len(matches) == 1:
            entity = matches[0]
            entity.count += 1
            if name not in entity.name_forms:
                entity.name_forms.append(name)
            locations = entity.locations
            if sentence_number in locations:
                locations[sentence_number] += index_list
            else:
                locations[sentence_number] = index_list
        # Otherwise make new entity
        else:
            entity = Entity(name, normalized_name, sentence_number=sentence_number, index_list=index_list)
            merged_entities.append(entity)

    return merged_entities


def normalize_name(name):
    '''
    Removes prefix and 's from the given name.
    Function from NU Infolab News Context Project (https://github.com/NUinfolab/context).
    '''
    name = name.split()
    for i, word in enumerate(name):
        if word.lower().strip().strip('.') not in NAME_PREFIXES:
            break
    no_prefix = name[i:]
    normalized = ' '.join(no_prefix)

    # When input text is unicode encoded in utf-8
    try:
        s = codecs.decode("’s", 'utf-8')
    except:
        s = "’s"
    if normalized.endswith("'s") or normalized.endswith(s):
        normalized = normalized[:-2]

    return normalized


def relevance_score(alpha, entity, num_sentences):
    '''
    Calculate the relevance score for the given entity.
    '''
    score = 0
    if entity.headline:
        score += alpha
    first_location = min([key for key in entity.locations]) + 1
    score += entity.count / (num_sentences * (first_location ** 0.25))
    return score


def select_high_score_entities(alpha, entity_list, num_sentences):
    '''
    Returns a list of the 3 entities with highest relevance score
    above a threshold of 0.07 (chosen after testing many articles).
    '''
    score_list = []
    for entity in entity_list:
        score = relevance_score(alpha, entity, num_sentences)
        score_list.append((entity, score))
        entity.relevance_score = score

    score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
    return [x[0] for x in score_list[:3] if x[1] > 0.07]  # threshold: 0.07


def get_headline_entities(headline, merged_entities):
    '''
    Extracts the entities from the headline and updates merged_entities accordingly.
    '''
    locations_found = {}
    tokens = word_tokenize(headline)
    for entity in merged_entities:
        for name in entity.name_forms:
            index_list = get_locations(name, tokens, locations_found)
            if index_list:
                count = len(index_list)
                # Replace to avoid double counting but maintain indeces
                for i in index_list:
                    if isinstance(i, int):
                        tokens[i] = ''
                    else:
                        for j in range(i[0], i[1]+1):
                            tokens[j] = ''
                entity.count += count
                entity.headline = True
                if entity.headline_locations:
                    entity.headline_locations += index_list
                else:
                    entity.headline_locations = index_list


def get_top_entities(headline, tokenized_article):
    '''
    Returns the top entities (as entity objects) from the given headline (string)
    and the tokenized article.
    '''
    temp_entities, num_sentences = extract_entities_article(tokenized_article)
    merged_entities_ = merge_entities(temp_entities)

    # Filter out images (entity recognizer thinks images are entities)
    merged_entities = []
    for entity in merged_entities_:
        if "image" not in entity.name.lower():
            merged_entities.append(entity)

    get_headline_entities(headline, merged_entities)
    top_entities = select_high_score_entities(0.01, merged_entities, num_sentences)
    return top_entities


'''
Filtered final draft of role dictionaries.
'''

HERO_DICT = {'gentle', 'preserving', 'leadership', 'amazing', 'devoted', 'humble', 'warned', 'surprised', 'humanity', 'brave', 'evacuate', 'redemption', 'smile', 'honor', 'revolutionize', 'leader', 'advocate', 'savior', 'charity', 'sympathies', 'kindness', 'good', 'protect', 'teach', 'reputation', 'respected', 'welfare', 'glory', 'victory', 'winner', 'well', 'contained', 'restoration', 'commitment', 'ability', 'efforts', 'inspire', 'safety', 'allies', 'health', 'strength', 'empowered', 'passion', 'encouraging', 'warm', 'vision', 'scored', 'authorities', 'justice', 'grand', 'admire', 'reshape', 'communities', 'response', 'strengthen', 'bolster', 'intervened', 'motivated', 'reconstruct', 'freedom', 'duty', 'aided', 'conquer', 'smart', 'bravery', 'improve', 'donate', 'wise', 'ingenuity', 'milestone', 'protections', 'expand', 'hero', 'pursuit', 'invent', 'containment', 'achievement', 'supporters'}

VILLAIN_DICT = {'contaminate', 'dirty', 'abduct', 'terror', 'worsen', 'crisis', 'lambast', 'abandonment', 'harass', 'subvert', 'virus', 'crime', 'provoke', 'kidnap', 'manipulate', 'alleged', 'refusal', 'trafficking', 'marginalize', 'conformity', 'clampdown', 'villain', 'disparaged', 'cold', 'exacerbate', 'alienate', 'commit', 'trial', 'violence', 'denounced', 'stripped', 'undermine', 'seize', 'persecuted', 'opposing', 'intimidate', 'jailed', 'fool', 'investigation', 'imprisoned', 'bias', 'deception', 'gunshots', 'threaten', 'hoax', 'engulfed', 'blame', 'eruption', 'offensive', 'contempt', 'suggested', 'coercion', 'erase', 'catastrophe', 'rumors', 'weaken', 'pointed', 'treason', 'evil', 'abused', 'sentenced', 'bullet', 'warn', 'devastate', 'convicted', 'rebuke', 'reveal', 'bully', 'collude'}

VICTIM_DICT = {'setback', 'injured', 'traumatized', 'prevented', 'healing', 'buried', 'stuck', 'anguished', 'flee', 'suffer', 'casualty', 'trampled', 'forsaken', 'harassed', 'harassment', 'hardship', 'deported', 'howling', 'shocked', 'violence', 'depressed', 'danger', 'mute', 'stripped', 'terrified', 'distrust', 'assassinated', 'shivering', 'sick', 'complain', 'abducted', 'huddled', 'victimized', 'persecuted', 'barricaded', 'devastated', 'kidnapped', 'seized', 'justified', 'evacuated', 'surrendered', 'diagnosed', 'imprisoned', 'independence', 'slave', 'deceased', 'rebuffed', 'target', 'trapped', 'screamed', 'loss', 'trafficked', 'humiliated', 'impairment', 'wounded', 'discriminated', 'disadvantaged', 'blood', 'offended', 'accuses', 'saddens', 'threatened', 'disaster', 'devastation', 'overshadowed', 'tortured', 'abused', 'remonstrated', 'jeopardizing', 'stabbed', 'prey', 'sentenced', 'challenged', 'renounced', 'scared', 'humiliation', 'deaths', 'rescued', 'bleeding'}


global namscore
namscore = []

# Parts of speech that are invalid in WordNet similarity function
IGNORE_POS = [
    "PRP",  # personal pronoun
    "PRP$",  # possessive pronoun
    "WP",  # wh-pronoun
    "WP$",  # possessive wh-pronoun
    "CC",  # coordinating conjunction
    "IN",  # preposition/subordinating conjunction
    "DT",  # determiner
    "WDT",  # wh-determiner
    "PDT",  # predeterminer
    "RP",  # particle
    "CD",  # cardinal digit
    "POS",  # possessive
    "UH",  # interjection
    "TO",  # to
    "LS",  # list marker
    "EX",  # existential there
    "FW",  # foreign word
]

# Constants to represent roles
HERO = 0
VILLAIN = 1
VICTIM = 2

# Spacy model for detecting active/passive entities
nlp = spacy.load('en')


# Raised when newspaper fails to extract article or headline
class NewspaperError(Exception):
    pass


def extract_by_newspaper(url):
    '''
    News article extraction from url using newspaper package.
    '''
    content = Article(url)
    content.download()
    content.parse()
    headline = content.title
    article = content.text
    return headline, article


@lru_cache(maxsize=1000000)
def word_similarity(word1, word2):
    '''
    Returns the Wu-Palmer similarity between the given words.
    Values range between 0 (least similar) and 1 (most similar).
    '''
    syns_w1 = wn.synsets(word1)
    syns_w2 = wn.synsets(word2)
    score = 0
    for w1 in syns_w1:
        for w2 in syns_w2:
            cur_score = w1.wup_similarity(w2)
            cur_score = w2.wup_similarity(w1) if not cur_score else cur_score
            if cur_score:
                score = max(score, cur_score)
    return score


def decay_function(decay_factor, entity_locations, term_index):
    '''
    Accounts for decay in score based on distance between words.
    '''
    minDist = float("inf")
    for loc in entity_locations:
        if isinstance(loc, int):
            distance = abs(term_index - loc)
        else:
            distance = min(abs(term_index - loc[0]), abs(term_index - loc[1]))
        minDist = min(distance, minDist)
    return (1 - decay_factor) ** minDist


def sentiment(word):
    '''
    Returns the sentiment of the given string as a float within
    the range [-1.0, 1.0].
    '''
    word_blob = TextBlob(word)
    return word_blob.sentiment.polarity


def choose_roles_by_sentiment(word):
    '''
    Uses the sentiment score of a term to determine which dictionaries are
    likely to be most useful. Negative sentiment maps to villain and victim
    dics, positive to hero, and neutral to all three.
    '''
    s = sentiment(word)
    if s > 0.15:
        return [HERO]
    elif s < -0.15:
        return [VILLAIN, VICTIM]
    else:
        return [HERO, VILLAIN, VICTIM]


@lru_cache(maxsize=1000000)
def similarity_to_role(word, role):
    '''
    Returns the similarity of the word to the role.
    '''
    # Check for preprocessed scores
    scores = SIM_DIC.get(word)
    if scores:
        score = scores[role]

    else:
        similarity_total, count_zero = 0, 0

        if role == HERO:
            dict_length = len(HERO_DICT)
            for hero_term in HERO_DICT:
                cur_score = word_similarity(word, hero_term)
                similarity_total += cur_score
                if cur_score == 0:
                    count_zero += 1

        elif role == VILLAIN:
            dict_length = len(VILLAIN_DICT)
            for villain_term in VILLAIN_DICT:
                cur_score = word_similarity(word, villain_term)
                similarity_total += cur_score
                if cur_score == 0:
                    count_zero += 1

        elif role == VICTIM:
            dict_length = len(VICTIM_DICT)
            for victim_term in VICTIM_DICT:
                cur_score = word_similarity(word, victim_term)
                similarity_total += cur_score
                if cur_score == 0:
                    count_zero += 1

        if dict_length - count_zero == 0:
            return 0

        score = similarity_total / (dict_length - count_zero)

    # Standardize scores (avg and sd from scores on top 10k english words)
    if role == HERO:
        avg = 0.3230
        std = 0.1239
    elif role == VILLAIN:
        avg = 0.2878
        std = 0.1202
    else:
        avg = 0.2901
        std = 0.1194

    return (score - avg) / std


def skip_word(word, pos):
    '''
    Returns true if the given word should be ignored in analysis.
    '''
    if any((
        len(word) < 3,
        pos in IGNORE_POS,
        word == "''",
        word == "``",
        word == '"',
    )):
        return True

    for stop in STOP_WORDS:
        if re.fullmatch(stop, word.lower()):
            return True

    return False


def active_passive_role(entity_string, sentence):
    '''
    Determine whether the entity is an active or passive role.
    Active roles = subject or passive object
    Passive roles = object or passive subject
    '''
    sent = nlp(sentence)
    is_active = False
    for tok in sent:
        if (tok.dep_ == "nsubj"):
            is_active = True
        if (str(tok) == entity_string):
            if (tok.dep_ == "nsubj"):
                return "active"
            if (tok.dep_ == "pobj" and not is_active):
                return "active"
            if (tok.dep_ == "pobj" and is_active):
                return "passive"
            elif (tok.dep_ == "dobj" or tok.dep_ == "nsubjpass"):
                return "passive"
            else:
                return "neutral"
    return "notInSentence"


def is_word_part_of_entity(entities_in_sent, sentence_index, word_index):
    '''
    Determine if the word at word_index is part of one of the top entities
    in the sentence.
    '''
    for entity in entities_in_sent:
        if sentence_index == "H":
            entity_locations = entity.headline_locations
        else:
            entity_locations = entity.locations[sentence_index]

        for loc in entity_locations:
            if isinstance(loc, int):
                if word_index == loc:
                    return True
            elif loc[0] <= word_index <= loc[1]:
                    return True
    return False


def get_top_words(word_dic):
    '''
    Returns a list of the top three (word, score) pairs in the dictionary.
    '''
    result = []
    for i, word in enumerate(sorted(word_dic, key=word_dic.get, reverse=True)):
        if i > 2:
            break
        result.append((word, word_dic[word]))
    print(result)
    #namscore = result.copy()
    #print(namscore)
    return result


def additional_score(act_pas, role, score):
    '''
    Computes an additional score if the active/passive input matches the
    given role. Active matches hero and villain. Passive matches victim.
    '''
    if act_pas == "active" and (role == HERO or role == VILLAIN):
        return score
    if act_pas == "passive" and role == VICTIM:
        return score
    return 0


def get_entities_act_pas(sentence_index, sentence, tokenized_sentence, entities_in_sent):
    '''
    Determines if each entity in the sentence is active or passive within the sentence.
    Returns the results in a list that corresponds to entities_in_sent.
    '''
    entities_act_pas = []
    for entity in entities_in_sent:
        last_loc = entity.locations[sentence_index][-1]  # Use last index of entity
        if isinstance(last_loc, int):
            entity_string = tokenized_sentence[last_loc]
        else:
            entity_string = tokenized_sentence[last_loc[1]]
        entities_act_pas.append(active_passive_role(entity_string, sentence))
    return entities_act_pas


# Global variables needed for functions below
top_entities, add_score, decay = 0, 0, 0
hero_scores, villain_scores, victim_scores, counts = [], [], [], []
top_hero_words, top_villain_words, top_victim_words = [{}], [{}], [{}]


def initialize_globals(entities, additional_score, decay_factor):
    '''
    Initialize entities, decay factor, and additional score for active/passive.
    Also set up lists for scores, counts, top words (indexed by entities).
    '''
    global top_entities, add_score, decay
    global hero_scores, villain_scores, victim_scores, counts
    global top_hero_words, top_villain_words, top_victim_words
    top_entities = entities
    add_score = additional_score
    decay = decay_factor
    n = len(entities)
    hero_scores, villain_scores, victim_scores, counts = ([0]*n for i in range(4))
    top_hero_words = [{} for _ in range(n)]
    top_villain_words = [{} for _ in range(n)]
    top_victim_words = [{} for _ in range(n)]


def get_entities_in_sent(sentence_index):
    '''
    Returns a list of entities in the sentence and updates counts.
    '''
    entities_in_sent = []
    for i, entity in enumerate(top_entities):
        if sentence_index in entity.locations:
            counts[i] += 1
            entities_in_sent.append(entity)
    return entities_in_sent


def update_total_scores(entity, role, word, word_score):
    '''
    Adds the given score to the entity's total score for the given role.
    Also updates top words for that role.
    '''
    entity_index = top_entities.index(entity)
    if role == HERO:
        hero_scores[entity_index] += word_score
        if word in top_hero_words[entity_index]:
            top_hero_words[entity_index][word.lower()] += word_score
        else:
            top_hero_words[entity_index][word.lower()] = word_score
    elif role == VILLAIN:
        villain_scores[entity_index] += word_score
        if word in top_villain_words[entity_index]:
            top_villain_words[entity_index][word.lower()] += word_score
        else:
            top_villain_words[entity_index][word.lower()] = word_score
    elif role == VICTIM:
        victim_scores[entity_index] += word_score
        if word in top_victim_words[entity_index]:
            top_victim_words[entity_index][word.lower()] += word_score
        else:
            top_victim_words[entity_index][word.lower()] = word_score


def score_word(word, word_index, sentence_index, entities_in_sent, entities_act_pas):
    '''
    Scores the word across the appropriate roles and updates totals.
    '''
    # Compute scores for roles that match word senitment
    roles_to_check = choose_roles_by_sentiment(word)
    scores = {}
    for role in roles_to_check:
        scores[role] = similarity_to_role(word, role)

    # Compute score for each entity-role pair and update totals
    for entity in entities_in_sent:
        for role in roles_to_check:
            cur_score = scores[role]
            act_pas = entities_act_pas[entities_in_sent.index(entity)]
            cur_score += additional_score(act_pas, role, add_score)
            cur_score *= decay_function(decay, entity.locations[sentence_index], word_index)
            update_total_scores(entity, role, word, cur_score)


def get_roles_and_top_words():
    entities_role_results = [(None, 0), (None, 0), (None, 0)]
    top_words = [None, None, None]
    namdict = []
    for i, entity in enumerate(top_entities):

        # Compute final role scores
        if counts[i] == 0:
            hero_score, villain_score, victim_score = 0, 0, 0
        else:
            hero_score = hero_scores[i] / counts[i]
            villain_score = villain_scores[i] / counts[i]
            victim_score = victim_scores[i] / counts[i]

        # Add relevance to scores
        hero_score += 5*entity.relevance_score
        villain_score += 5*entity.relevance_score
        victim_score += 5*entity.relevance_score

        # Determine entity role based on max role score
        sorted_scores = sorted([hero_score, villain_score, victim_score], reverse=True)
        max_score = sorted_scores[0]
        if max_score - sorted_scores[1] >= 0.05:  # Don't assign if scores too close
            if hero_score == max_score:
                entity.role = HERO
            elif villain_score == max_score:
                entity.role = VILLAIN
            elif victim_score == max_score:
                entity.role = VICTIM

        # Assign entity to corresponding role if score is larger than current leader
        if entity.role is not None and max_score > entities_role_results[entity.role][1]:
            entities_role_results[entity.role] = (entity.name, max_score)

            # Update top words accordingly
            if entity.role == HERO:
                top_words[HERO] = [x[0] for x in get_top_words(top_hero_words[i])]
                namdict.append(get_top_words(top_hero_words[i]))
            elif entity.role == VILLAIN:
                top_words[VILLAIN] = [x[0] for x in get_top_words(top_villain_words[i])]
                namdict.append(get_top_words(top_hero_words[i]))
            else:
                top_words[VICTIM] = [x[0] for x in get_top_words(top_victim_words[i])]
                namdict.append(get_top_words(top_hero_words[i]))

    result = [x[0] for x in entities_role_results]
    print(result)
    print(top_words)
    print(namdict)
    #print(namscore)
    return result, top_words,namdict


def assign_roles(url, add_score, decay_factor):
    # Extract article from url and tokenize
    try:
        headline, article = extract_by_newspaper(url)
    except:
        raise NewspaperError
    if len(article) == 0:
        raise NewspaperError
    tokenized_article = sent_tokenize(article)

    # Remove headline from article if it's added to first sentence (common extraction error)
    if (headline in tokenized_article[0]):
        tokenized_article[0] = tokenized_article[0].replace(headline, "")

    # Get top entities and initialize globals
    entities = get_top_entities(headline, tokenized_article)
    initialize_globals(entities, add_score, decay_factor)

    # Loop through each sentence
    for sentence_index in range(len(tokenized_article)):

        # Find entities in sentence and update counts
        entities_in_sent = get_entities_in_sent(sentence_index)

        # Skip sentence if no entities.
        if not entities_in_sent:
            continue

        # Get and tokenize sentence
        sentence = tokenized_article[sentence_index].strip()
        tokenized_sentence = word_tokenize(sentence)

        # Compute active/passive for each entity in sentence
        entities_act_pas = get_entities_act_pas(sentence_index, sentence, tokenized_sentence, entities_in_sent)

        # Loop through words in sentence
        tagged_sentence = pos_tag(tokenized_sentence)
        for word_index in range(len(tokenized_sentence)):

            # Skip word if it is part of an entity
            if is_word_part_of_entity(entities_in_sent, sentence_index, word_index):
                continue

            # Check if word is a skip word (stop words, invalid POS, punctuation)
            word = tokenized_sentence[word_index]
            pos = tagged_sentence[word_index][1]
            if skip_word(word, pos):
                continue

            # Score word and track results
            score_word(word, word_index, sentence_index, entities_in_sent, entities_act_pas)

    # Finalize assignment and top words
    return get_roles_and_top_words()


# Allow cross-origin resource sharing
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Run role assignment algorithm
@app.route('/')
def roleassign():
    url = request.args.get("url")
    try:
        role_names, top_words, namdict= assign_roles(url, 2.0, 0.25)
        top_entity_names = []
        for name in role_names:
            name_string = name if name else "None"
            top_entity_names.append(name_string)
        print("aaaaaaaaaaaaaaaaaaaaa")
        print(top_entity_names, top_words,namdict)
        return jsonify(top_entity_names, top_words)
    except NewspaperError:
        return jsonify("Extraction error")
    except:
        return jsonify("Entity recognition/role assignment errors")


# if __name__ == '__main__':
#     app.run(debug=True)
    
#assign_roles("https://www.theguardian.com/world/2021/mar/22/okinawa-us-airbase-soil-war-dead-soldiers-japan",2.0, 0.25)