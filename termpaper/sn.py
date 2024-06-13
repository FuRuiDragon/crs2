# from transformers import pipeline
#
# sentiment_analysis = pipeline("sentiment-analysis", model="EleutherAI/gpt-neo-2.7B")
#
# result1 = sentiment_analysis("This is a bad day!")
# result2 = sentiment_analysis("This is an amazing day!")
# result3 = sentiment_analysis("This is an usual day!")
#
# print(result1)
# print(result2)
# print(result3)



# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")


# TODO: to determin wether the topic of term paper is interesting, feasible/correct/okay/proper, not feasible/proper
# sentiment analysis, aka. opinion mining or emothion AI, on emotional tone convey by the writer, using NLP and linguistic rules to identify the sentiment expressed in the text
# Sentiment analysis is often classified into three categories: positive, neutral, and negative.

# steps:
# 1. get GPT-4 API key
# 2. prepare our dataset in order to fine-tune the GPT-4 model, https://promptly.engineering/resources/how-to-use-gpt-4-for-sentiment-analysis

## alternatively, use GPT for labelling ? or using label studio for labelling, https://labelstud.io/


import openai
import re


openai.api_key = "your openai api key"


# to identify whether a student is sure about the topic he supposed or not
def identify_stu_confidence(student_tp_text):

    prompt = f""" 
        Perceive the confidence levels, when authors are writing their proposed  research topics and research questions, for example, 
        when an author says he is not sure about something, which is he has low confidence; when an author talks a lot about something, 
        he is kinda of have confidence; when an author can not say something specifically, then his confidence is neutral.
        
        Now Classify the sentiment of the following text as 'positive', 'negative', or 'neutral': <input_text>

        Respond in the json format: {{'response': sentiment_classification}}\nText: {student_tp_text}\nSentiment (positive, neutral, negative):

        """

    return prompt


def identify_topic_feasibility(input_text):
    prompt = f"""
    Classify the sentiment of the following text as 'positive', 'negative', or 'neutral': <input_text>

    Respond in the json format: {{'response': sentiment_classification}}\nText: {input_text}\nSentiment (positive, neutral, negative):

    """
    return prompt


def get_sentiment_gpt(prompt, input_text):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=40,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response_text = response.choices[0].message['content'].strip()
    sentiment = re.search("negative|neutral|positive", response_text).group(0)
    # Add input_text back in for the result
    return {"text": input_text[:25] + " ...", "response": sentiment}


# Test with student termpaper examples

stu_tp_text1= f""" Following initial literature research, I would like to deal with the topics of learning analytics and gamification in this term paper.

- How can gamification be supported by learning analytics?

- What advantages can data collected during learning processes with gamification offer?

- Does gamification lead to increased motivation to learn?"""


stu_tp_text2=f""" My proposed topic: "Marginalization and dysmorphia through media - the process of othering in old media and new educational technologies" 
Research question: What is the significance of generating a positive imaginary in a digital society?
Further research questions: 
To what extent were/are media used for purposes of discrimination and marginalization?
To what extent can socialization under the circumstances of marginalization lead to distortions (dysmorphia) of identity construction?
What emancipatory educational opportunities are offered to subalterns in the net cultures of the media world?
What methods do subalterns use to optimize their connectivity in the digital society?
Which media-specific strategies are used from an Afrofuturist perspective to deconstruct the category of marginalization?
To what extent can Afrofuturism be made fruitful for media education as a form of emancipatory educational technology?
To what extent do media education programs need to be reconsidered in educational practice in order to remain attractive to a young media-active subaltern (living in Germany)?"""

stu_tp_text3=f"""This raises the question of the didactic quality of digital learning opportunities. This would be an interesting question for my term paper, but I'm not yet sure whether research results and literature are available.

Another interesting question for me is how adult education providers are dealing with increasing digitalization and mediatization. What is the current status? How do adult education providers have to change in order to keep pace and what are the conditions for the successful implementation of media didactic concepts? """


stu_tp_text4 =f""" What are the advantages and disadvantages of using gamification in dual vocational training?
What are the advantages and disadvantages of using gamification in continuing education in the field of XXX? """

prompt1 = identify_stu_confidence(stu_tp_text1)
sentiment1 =get_sentiment_gpt(prompt1, stu_tp_text1)
print("Result\n",f"{sentiment1['response']}")

prompt2 = identify_stu_confidence(stu_tp_text2)
sentiment2 =get_sentiment_gpt(prompt2, stu_tp_text2)
print("Result\n",f"{sentiment2}")

prompt3 = identify_stu_confidence(stu_tp_text3)
sentiment3 =get_sentiment_gpt(prompt1, stu_tp_text3)
print("Result\n",f"{sentiment3}")

prompt4 = identify_stu_confidence(stu_tp_text4)
sentiment4 =get_sentiment_gpt(prompt1, stu_tp_text4)
print("Result\n",f"{sentiment4}")


# Test with teacher feedback examples

teacher_feedback = f"""I can now welcome you personally to the new semester. Thank you very much for your thoughts.

With regard to your choice of topic, I can say that you address areas that are both interesting and relevant.

However, we should bear in mind that the thesis should only be 15 pages long. This means: initial narrowing down is necessary! I would suggest, for example, that we could look at the phenomenon of othering using a selected educational technology.

What do you think?"""

teacher_feedback = f""" Your suggestions for topics are also worth considering: it would be conceivable to analyze a pedagogical scenario in terms of media didactics or to reflect on digitalization or mediatization for adult education (perhaps narrowing it down by organizational structural aspects).

The decision would be entirely up to you!"""

teacher_feedback = f"""It looks like you have not studied the course module 3A very well. The topic you suggested is not so proper. However, you should bear in mind that the thesis should only be 15 pages long. This means: initial narrowing down is necessary! I would suggest, for example, that we could look at the phenomenon of othering using a selected educational technology.
It looks like you have not studied the course module 3A very well. The topic you suggested is not so proper. However, you should bear in mind that the thesis should only be 15 pages long. This means: initial narrowing down is necessary! I would suggest, for example, that we could look at the phenomenon of othering using a selected educational technology.
 """

t_prompt = identify_topic_feasibility(teacher_feedback)
sentiment = get_sentiment_gpt(t_prompt, teacher_feedback)
print("Result\n",f"{sentiment}")
