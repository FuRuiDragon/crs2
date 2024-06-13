import json
import openai
from collections import defaultdict
import gpts.prompts
import gpts.test_gpt
from itertools import product
from itertools import combinations
import random

import os
#print(os.path.dirname(os.getcwd()))
#print(os.path.dirname(__file__))

# item_corpus_file = os.path.join(os.path.dirname(__file__), '../data/gpt35_autoGene/itemCorpus.json')
# concept_refs_file = os.path.join(os.path.dirname(__file__), '../data/gpt35_autoGene/conceptRefs.json')
delimiter = "####"


# create model_file pathes.
# filename, either "itemCorpus.json" or "conceptRefs.json"
def create_path(modelName, filename):
    path = os.path.dirname(os.getcwd())
    print(path)
    if "3.5" in modelName:
        modelName = "../data/gpt35_autoGene/" + filename
    elif "4" in modelName:
        modelName = 'data/gpt40_autoGene/' + filename
    elif "llama2" in modelName:
        modelName = "../data/llama2_autoGene/" + filename
    else:
        return None
    print(modelName)
    filepath = os.path.join(path, modelName)
    print(filepath)
    return os.path.join(os.path.dirname(os.getcwd()), modelName)


# ask GPT for topics/subjuects of course Media Education as categories
def create_category_message():
    system_message_content = f"""
            You are a teacher of the course "Medienpädagogik und Medienkommunikation" .\
            The most recent user query will be delimited with \
            {delimiter} characters.
            Output a python list of objects, where each object has \
            the following format:
                'category': <list of topics or subject>,
                'definition': <text to define the theme>

            Only output the list of objects, with nothing else.
            """

    user_message_content = f""" 
           Please list of the content what I should study. \
            """

    combined_messages = gpts.prompts.customize_prompt(system_message_content, user_message_content, "")
    print("sys+user messages: ", combined_messages)
    return combined_messages


# ask GPT for reference lists to a specific concept
def create_ref_message(concepts):
    # concept = "Media Literacy"
    # concept1 = "Digital Citizenship"
    system_message_content = f"""
               You are a expert on the topic of "Medienpädagogik und Medienkommunikation" .\
               The most recent user query will be delimited with \
               {delimiter} characters.
               
               Output a python list of objects (max 3 most influential of references or literatures of  \
               research papers or articles) from journal or conferences or books, where each object has \
               the following format:
                   'Title': <title of the research paper or article>,
                   'Authors': <list of the authors>,
                   'Source': <Journal or Conference or Book>,
                   'Date': <month, year. Both if possible>,
                   'Summary': <summary of the paper or article>   

               Only output the list of objects, with nothing else.
               """

    user_message_content = f""" 
              Can you please provide me some literatures which covers all of {delimiter}{concepts}{delimiter} 
               """

    combined_messages = gpts.prompts.customize_prompt(system_message_content, user_message_content, "")
    print("sys+user messages: ", combined_messages)
    return combined_messages


# ask GPT for answers and save response to files
def get_gpt_response(messages, modelname, filename):
    if not modelname or "4" not in modelname:
        modelname = "gpt-3.5-turbo"
    response = gpts.test_gpt.get_completion0(messages, modelname)
    print("topics of MM: ", response['text'])

    ctext = response['text'].replace('\n', '').replace('"', '').replace('\'', '"')
    json_object = json.loads(ctext)
    # print("type(json_object) is: ", type(json_object)) # type(json_object) is:  <class 'list'>

    with open(filename, "w") as write_file:
        json.dump(json_object, write_file, indent=4)
    return None


# * automatically generate item corpus with a given GPT model
def auto_generate_itemCorpus(modelname):
    combined_mass = create_category_message()
    item_corpus_file = create_path(modelname, "itemCorpus.json")
    get_gpt_response(combined_mass, modelname, item_corpus_file)
    return None


## below for retrieving references

# get references of one concept
def get_concept_ref(concepts_string, modelName):
    combined_mass = create_ref_message(concepts_string)
    # get_gpt_response(combined_mass, concept_refs_file)
    if not modelName or "4" not in modelName:
        modelName = "gpt-3.5-turbo"
    print('model: ', modelName)
    response = gpts.test_gpt.get_completion0(combined_mass, modelName)

    print("topics of MM: ", response['text'])

    ctext = response['text'].replace('\n', '').replace('"', '').replace('\'', '"')
    json_object = json.loads(ctext)
    return json_object


# * automatically generate references with a given GPT model
def auto_generate_references(modelName):
    all_refers = []
    # # since gpt-3.5 does not allow to be used for retrieving a big amount of data,
    # # sampled_list = random.sample(concepts_list, 10)

    if "3.5" in modelName:
        concepts_list = ['Media Education', 'Media Communication Theories', 'Mass Communication',
                     'Interpersonal Communication',
                     'Media Literacy', 'Digital Citizenship', 'Media and Society', 'Media Production', 'Media Ethics',
                     'Media Law',
                     'Media Regulation']
    else:
        concepts_list = get_conceptList(modelName)

    # for concepts in sampled_list:
    for concepts in concepts_list:
        # resultStrs = ','.join(map(str, concepts))
        resultStrs = concepts
        # print('resultString is: ', resultStrs)
        refer_list = get_concept_ref(resultStrs, modelName)
        temp_dic = {"concepts": resultStrs, "references": refer_list}
        print("temp_dic is: ", temp_dic)
        all_refers.append(temp_dic)

    print("all refers are: ", all_refers)

    concept_refs_file = create_path(modelName, "conceptRefs.json")
    print(concept_refs_file)
    with open(concept_refs_file, "w") as write_file:
        json.dump(all_refers, write_file, indent=4)
    return None


# get concept list from ontology, i.e., from file data/origin/derived/concept_de.json or media_concept_en.json
def get_OntoConcepts():
    list_category =[]
    all_concepts = []
    filename = os.path.join(os.path.dirname(__file__), '../data/origin/derived/media_concept_en.json')
    # Opening JSON file
    with open(filename) as json_file:
        data = json.load(json_file)

        # for reading nested data [0] represents
        # the index value of the list
        # print(type(data))# list
        for x in data:
            # for printing the key-value pair of
            # nested dictionary for loop can be used
            #print(x)

            category_concepts = {}
            category_concepts["category"] = x['category']
            temp_list = []
            for i in x['concepts']:
                temp_list.append(i['concept'])
                all_concepts.append(i['concept'])
            category_concepts["concepts"]= temp_list
            list_category.append(category_concepts)
            print(category_concepts)
        print(len(list_category), " categories, ", len(all_concepts), " concepts are retrieved from domain Ontology")
    return list_category, all_concepts



# === get all category names from the file "itemCorpus.json"
def get_conceptList(modelName):
    # 1. read and load concepts from .json
    num_combination = 4
    #filename = os.path.join(os.path.dirname(__file__), '../data/gpt35_autoGene/itemCorpus1.json')
    filename = create_path(modelName, 'itemCorpus.json')
    with open(filename, 'r') as f:
        response = f.read()
        response = response.replace('\n', '')
        response = response.replace('}{', '},{')
        if not response.startswith("["):
            response = "[" + response
        if not response.endswith("]"):
            response = response + "]"
        data = json.loads(response)
    # print('data is', data)
    print('type(data): ', type(data))  # list
    concept_list = []
    for obj in data:
        concept_list.extend(obj['category'])
    print('concept_list :', concept_list)

    # 2. get combination of concept list, max 4 concepts as one search string
    concepts_list = get_combined(concept_list, num_combination)

    #  3. for each concept to get a list of  references

    return concepts_list


def combine(temp_list, m):
    temp_list2 = []
    for c in combinations(temp_list, m):
        temp_list2.append(c)
    return temp_list2


# only limited 4 combined concepts as a search string
def get_combined(user_list, n):
    end_list = []
    # for i in range(len(user_list)):
    for i in range(n + 1):
        end_list.extend(combine(user_list, i))
    print("end_list :", end_list)
    return end_list

# ==============


if __name__ == '__main__':
    model35="gpt-3.5-turbo"
    model4 = "gpt-4"
    #auto_generate_itemCorpus(model35)
    #auto_generate_references(model35)

    auto_generate_itemCorpus(model4)
    auto_generate_references(model4)

    # get_OntoConcepts()
