import openai
import gpts.test_gpt as gpts
import termpaper.utils as tpu
import termpaper.sn as sn
import json
from flask import Flask, g


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def ask_gpt_directly(user_text):
    system_message = f"""
            You are an expert on media education and media communication. \
            Respond in a friendly and helpful tone in English, with VERY concise answers. 
        """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_text}
    ]
    response = gpts.get_completion0(messages)
    return response


# to have one chat, one round conversation from user and to user
def chats():

    return None

# to give feedbacks to Term Paper: System of chained prompts for processing the user query
def process_stu_message(user_input, all_messages, debug=True):
    delimiter = "```"
    total_tokens = 0
    chain_of_thoughts = ""

    # Step 0: Check input to see if it flags the Moderation API or is a prompt injection
    response = openai.Moderation.create(input=user_input)
    moderation_output = response["results"][0]

    if moderation_output["flagged"]:
        str_step1 = "Step 1: Input flagged by OpenAI Moderation API."
        print(str_step1)
        chain_of_thoughts += str_step1
        return "Sorry, we cannot process this request."

    if debug: print("Step 0: Input passed moderation check.")
    chain_of_thoughts += "Step 0: Input passed moderation check.\n"

    # Step 1: to identify whether a student is sure about the topic he has supposed or not
    sn_prompt = sn.identify_stu_confidence(user_input)
    confidence_sn = sn.get_sentiment_gpt(sn_prompt, user_input)
    print("Result\n", f"{confidence_sn}")

    if confidence_sn['response'] == 'positive':
        # chat massage appended
        print('We are happy to see that you are confident with your research topic and we will help you on this topic..')

    if confidence_sn['response'] == 'negative':
        # chat massage appended
        print('You seem not sure about the research topic and then we would suggest you a new topic from an expert perspective. Will be okay with you?')

    if confidence_sn['response'] == 'neutral':
        # chat massage appended
        print('You seem you are not so excited with the research topic. Are your open to suggestion to a new topic from an expert perspective?')


    #  with sys_msg#1 and a given list of category/topics of MediaEducation course module,
    #  Step 2: to get the first results of category and topics relating to student's idea of term paper
    topics = tpu.get_topic_list()
    # print("topics: ", topics)
    # print("topics is empty? ", len(topics) ) # 13

    subjects_concepts_response = tpu.find_topics(user_input, topics)

    if debug: print("Step 2: grab some topics or subjects from Student's ideas.")
    chain_of_thoughts += "\nStep 2: grab some topics or subjects from Student's ideas.\n"
    chain_of_thoughts += "\n" + "<pre>" + subjects_concepts_response + "</pre>" + "\n"

    print("\nstep 2. results on subjects:\n", subjects_concepts_response)
    print("\n\ntype of step2_result: ", type(subjects_concepts_response))  # <class 'list'>

    if not is_json(subjects_concepts_response):
        return ask_gpt_directly(user_input), []
    else:
        # Step 3: recommend the list of references based on the topics/subjects from student text,
        # from external data files/db/knowledge graph

        # convert <class 'str'> to list
        result_1 = json.loads(subjects_concepts_response)
        # print("\n\ntype of step2_result: ", type(result_1))  # <class 'list'>
        print("step2_result is:", result_1)

        # for each subject to recommend N references
        N = 2
        print("result_1: ", result_1)
        print(type(result_1))
        for ele in result_1:
            # print("ele is:", ele)
            related_concepts = ele['related_categories']
            result_2, no_found_references = tpu.find_references(related_concepts)
            refer_N = result_2[:N]
            ele['recommended_references'] = refer_N

        # Step 3: If products are found, look them up
        if debug: print("Step 3: Looked up subject references information.")
        print("\nStep 3 results on recommended references:\n", result_1)
        chain_of_thoughts += "\nStep 3: Looked up subject references information.\n"
        chain_of_thoughts += "\n " + str(
            no_found_references) + " references found in all. \n\n The top 2 will be recommended.\n\n"
        # chain_of_thoughts += "\n" + result_1 + "\n"
        chain_of_thoughts += "<pre>" + json.dumps(result_1, indent=4).replace(' ', '&nbsp').replace(',\n',
                                                                                                    ',<br>').replace(
            '\n', '<br>') + "</pre>"

        # Step 4: Answer the user question
        system_message = f"""
        You are a knowledge expert. \
        Respond in a friendly， analysis and recommendation tone, with concise answers in English. \
        Make sure to ask the user relevant follow-up questions.
        """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
            {'role': 'assistant', 'content': f"Relevant product information:\n{result_1}"}
        ]

        final_response = gpts.get_completion_from_messages(all_messages + messages)
        if debug: print("\n\nStep 4: Generated response to user question.")
        chain_of_thoughts += "\nStep 4: Generated response to user question."

        all_messages = all_messages + messages[1:]

        print("\nStep 4 result is:\n", final_response)

        # Step 5: Put the answer through the Moderation API
        response = openai.Moderation.create(input=final_response['text'])
        moderation_output = response["results"][0]

        if moderation_output["flagged"]:
            if debug: print("Step 5: Response flagged by Moderation API.")
            chain_of_thoughts += "\n\nStep 5: Response flagged by Moderation API."
            return "Sorry, we cannot provide this information."

        if debug: print("\nStep 5: Response passed moderation check.")
        chain_of_thoughts += "\n\nStep 5: Response passed moderation check."

        # Step 6: Ask the model if the response answers the initial user query well
        user_message = f"""
        Customer message: {delimiter}{user_input}{delimiter}
        Agent response: {delimiter}{final_response}{delimiter}
    
        Does the response sufficiently answer the question?
        """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]
        evaluation_response = gpts.get_completion_from_messages(messages)
        print("\n evaluation_response: ", evaluation_response)
        if debug: print("\nStep 6: Model evaluated the response.")
        chain_of_thoughts += "\nStep 6: Model evaluated the response."

        # Step 7: If yes, use this answer; if not, say that you will connect the user to a human
        if "Y" in evaluation_response:  # Using "in" instead of "==" to be safer for model output variation (e.g., "Y." or "Yes")
            if debug: print("\nStep 7: Model approved the response.")
            chain_of_thoughts += "\nStep 7: Model approved the response."
            final_response["chain_of_thoughts"] = chain_of_thoughts
            return final_response, all_messages
        else:
            if debug: print("\nStep 7: Model disapproved the response.")
            chain_of_thoughts += "\nStep 7: Model disapproved the response."
            neg_str = "I'm unable to provide the information you're looking for. I'll connect you with a human \
                      representative for further assistance. "
            # return  neg_str, all_messages
            final_response["chain_of_thoughts"] = chain_of_thoughts
            return final_response, all_messages
