import os, json
import numpy as np
from sklearn.cluster import HDBSCAN
from collections import defaultdict
import pypdfium2 as pdfium
from transformers import pipeline
from langchain import hub
from langchain.schema import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint

# UniEval model functions
from utils import convert_to_json
from metric.evaluator import get_evaluator

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_oTHdfvrbAgZDD...PHfTaklKWXycp"

label = pipeline(task="text-classification", model="lighteternal/fact-or-opinion-xlmr-el")
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",temperature=0.01)

claim_generation_prompt = hub.pull("murt147/claim-generation-prompt")
claim_compare_prompt = hub.pull("murt147/claim-compare-prompt")
claim_compare_edge_case_prompt = hub.pull("murt147/claim-compare-edge-case-prompt")
key_word_generation_prompt = hub.pull("murt147/key_word_generation_prompt")
inference_query_prompt = hub.pull("murt147/inference-query-prompt")

global_evidence_threshold = 0.6

def get_text_func(pdf):
    """
    Gets the text from a pdf file using pypdfium2 functions.

    Args: name of the pdf file as a string
    Ret: text from the pdf file as a string
    """
    text = ""
    pdf_reader = pdfium.PdfDocument(pdf)
    for i in range(len(pdf_reader)):
        page = pdf_reader.get_page(i)
        textpage = page.get_textpage()
        text += textpage.get_text_bounded() + "\n"
    return text

def get_chunks_func(text):
    """
    Creates chunks from text.

    Args: text as a string
    Ret: chunked text as a list, each element of the list is a chunk of text
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def evidence_search(pdf):
    """
    Compiles a list of facts/objective statements from a string of text.

    Args: text as a string
    Ret: a list of facts/objective statements from the text, each element in the list is a fact/objective statement
    """
    text = get_text_func(pdf=pdf)
    chunks = get_chunks_func(text=text) # consider this as at least 2+ chunks
    labelled_chunks = label(chunks) # labels each chunk as 0 or 1, 1 for facts/objective statements, 0 otherwise

    sorted_label_chunks = []
    fact_chunks = []
    for i in range(len(labelled_chunks)):
        if labelled_chunks[i].get('label') == "LABEL_1": # score should be the label_1 score
            sorted_label_chunks.append({'text':chunks[i],'score':labelled_chunks[i].get('score')}) 

    # sort sorted label chunks list, greatest score is first element
    sorted_label_chunks = sorted(sorted_label_chunks, key=lambda item: item['score'], reverse=True)

    ctr = 0
    while(ctr < len(sorted_label_chunks)):
        if len(fact_chunks) < 2:
            fact_chunks.append(sorted_label_chunks[ctr]['text'])
            ctr+=1
        elif sorted_label_chunks[ctr]['score'] > global_evidence_threshold:
            fact_chunks.append(sorted_label_chunks[ctr]['text'])
            ctr+=1
        else:
            break
    
    return fact_chunks

def claim_check_func(src, output):
    """
    Checks if a claim is in alignment with the evidence used to generate it.

    Args: evidence and claim, as a string
    Ret: true if claim aligns with evidence, false otherwise
    """
    task = 'fact'
    src_list = [src] # evidence
    output_list = [output] # claim
    
    data = convert_to_json(output_list=output_list, src_list=src_list) # Prepare data for pre-trained evaluators
    evaluator = get_evaluator(task) # Initialize evaluator for a specific task
    eval_scores = evaluator.evaluate(data, print_result=False) # Get factual consistency scores

    return eval_scores[0]["consistency"]

def generate_claim(evidence, temperature):
    """
    Generates a single claim based on a piece of evidence.

    Args: fact/objective statement as a string and the llm temperature
    Ret: generated claim and all its relevant details (evidence used, the claim itself, claim target, and claim topic), as a string
    """
    llm.temperature = temperature
    claim_gen_executor = (claim_generation_prompt|llm|StrOutputParser())
    claim = claim_gen_executor.invoke({"input": evidence})
    return claim

def create_data_list(evidence_list):
    """
    Creates a list of claims will all relevant details (evidence, claim, claim target, claim topic).

    Args: A list of strings representing a list of evidence chunks
    Ret: A list of dictionaries/hashmaps representing a list of unverified claims, each relevant detail is its own key
    """
    data_list = []
    
    # if the claim generated does not contain all relevant details, generate the claim again but increase the temperature of the llm until all relevant details are included.
    for i in range(len(evidence_list)):
        temperature = 0.01
        output = generate_claim(evidence_list[i], temperature)
        while not ("evidence" in output and "claim" in output and "claim target" in output and "claim topic" in output) and temperature <= 1:
            temperature+=0.01
            output = generate_claim(evidence_list[i], temperature)
        
        # Parse the llm output so that it only contains the json blob text
        output = output[output.index("```json")+7:]
        output = output[:output.index("```")]
        
        # Parse the json blob string into a dictionary/hashmap
        try:
            data = json.loads(output)
            data_list.append(data)
        except Exception as e:
            print(e)
            continue
    return data_list

def create_verified_list(raw_data_list):
    """
    Returns a list of verified claims by filtering a list of unverified claims. A verified claim is aligned with the evidence used to generate it.

    Args: list of dictionaries/hashmaps representing unverified claims
    Ret: list of dictionaries/hashmaps representing verified claims
    """
    sorted_data_list = []
    verified_data_list = []

    for i in range(len(raw_data_list)):
        sorted_data_list.append({'text':raw_data_list[i],'score':claim_check_func(src=raw_data_list[i]["evidence"], output=raw_data_list[i]["claim"])})
    
    # sort sorted data list
    sorted_data_list = sorted(sorted_data_list, key=lambda item: item['score'], reverse=True)

    ctr = 0
    while(ctr < len(sorted_data_list)):
        if len(verified_data_list) < 2:
            verified_data_list.append(sorted_data_list[ctr]['text'])
            ctr+=1
        elif sorted_data_list[ctr]['score'] > global_evidence_threshold:
            verified_data_list.append(sorted_data_list[ctr]['text'])
            ctr+=1
        else:
            break
    
    return verified_data_list

def compare_func(data1, data2, temperature):
    """
    Compares the similarity (of bridge entity and bridge topic) between two claims.

    Args: Two dictionaries/hashmaps, each representing a claim and all its relevant details (evidence used, the claim itself, claim target, and claim topic) and the llm temperature
    Ret: A string representing a dictionary with both claim targets and topics, and a score 1 if claims are similar, 0 otherwise
    """
    llm.temperature = temperature
    claim_compare_executor = (claim_compare_prompt|llm|StrOutputParser())
    value = claim_compare_executor.invoke({'input': [data1, data2]})
    return value

def cluster_func(similarity_matrix):
    hdb = HDBSCAN(min_cluster_size=2, max_cluster_size=4)
    hdb.fit(similarity_matrix)
    labels =  hdb.labels_

    clusters = defaultdict(list)
    for i in range(len(labels)):
        if labels[i] >= 0:
            clusters[labels[i]].append(i)
    
    print(labels)
    print(list(clusters.values()))

    return list(clusters.values())

def cluster_edge_case_func(verified_data_list, temperature):
    """
    Returns the two most similar elements in a list of claims.

    Args: A list of dictionaries/hashmaps, each representing a claim and all its relevant details (evidence used, the claim itself, claim target, and claim topic) and the llm temperature
    Ret: The indices of the two most similar elements in the list
    """
    llm.temperature = temperature
    cluster_edge_case_executor = (claim_compare_edge_case_prompt|llm|StrOutputParser())
    indices = cluster_edge_case_executor.invoke({'input': verified_data_list})
    return indices

def create_claim_sets(verified_data_list):
    # create 2d array of indices from the verified data list
    n = len(verified_data_list)
    similarity_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1
            else:
                temperature = 0.01
                output = compare_func(verified_data_list[i], verified_data_list[j], temperature)
                while not ("claim target 1" in output and "claim topic 1" in output and "claim target 2" in output and "claim topic 2" in output and "score" in output) and temperature <= 1:
                    temperature+=0.01
                    output = compare_func(verified_data_list[i], verified_data_list[j], temperature)
        
                # Parse the llm output so that it only contains the json blob text
                output = output[output.index("```json")+7:]
                output = output[:output.index("```")]
                # print(output)

                # Parse the json blob string into a dictionary/hashmap
                try:
                    data = json.loads(output)
                    similarity_matrix[i, j] = data.get("score")
                except AttributeError:
                    similarity_matrix[i, j] = data[0].get("score")
                    continue
                except Exception as e:
                    print(e)
                    continue
    
    print(similarity_matrix)
    clusters = cluster_func(similarity_matrix) # hierarchical [[0,1,3],[2]]

    claim_sets = []
    if len(clusters) >= 2:
        for i in range(len(clusters)):
            claim_set = []
            for j in range(len(clusters[i])):
                claim_set.append(verified_data_list[clusters[i][j]])
            claim_sets.append(claim_set)
    else:
        claim_topic_list = []
        for i in range(len(verified_data_list)):
            claim_topic_list.append(verified_data_list[i].get("claim topic"))

        temperature = 0.01
        output = cluster_edge_case_func(claim_topic_list, temperature)
        while not ("element 1" in output and "element 2" in output and "indices" in output) and temperature <= 1:
            temperature+=0.01
            output = cluster_edge_case_func(claim_topic_list, temperature)
        
        # Parse the llm output so that it only contains the json blob text
        output = output[output.index("```json")+7:]
        output = output[:output.index("```")]

        # Parse the json blob string into a dictionary/hashmap
        try:
            data = json.loads(output)
            indices = data.get("indices")
            claim_sets.append(verified_data_list[indices[0]])
            claim_sets.append(verified_data_list[indices[1]])
        except Exception as e:
            print(e)

    return claim_sets

def generate_key_word(evidence, temperature):
    """
    Generates a single keyword based on a piece of evidence.

    Args: fact/objective statement as a string and the llm temperature
    Ret: a keyword and the evidence used, as a string
    """
    llm.temperature = temperature
    key_word_gen_executor = (key_word_generation_prompt|llm|StrOutputParser())
    key_word = key_word_gen_executor.invoke({"input": evidence})
    return key_word

def get_key_words(claim_sets):
    key_words_total = []
    for i in range(len(claim_sets)):
        key_words_subset = []
        for j in range(len(claim_sets[i])):
            temperature = 0.01
            output = generate_key_word(claim_sets[i][j].get("evidence"), temperature)
            while not ("evidence" in output and "keyword" in output) and temperature <= 1:
                temperature += 0.01
                output = generate_key_word(claim_sets[i][j].get("evidence"), temperature)
            
            # Parse the llm output so that it only contains the json blob text
            output = output[output.index("```json")+7:]
            output = output[:output.index("```")]
            # print(output)

            # Parse the json blob string into a dictionary/hashmap
            try:
                data = json.loads(output)
                key_word = data.get("keyword")
            except Exception as e:
                print(e)
                continue

            key_words_subset.append(key_word)
        key_words_total.append(key_words_subset)
    return key_words_total

def generate_inference_query(claims, key_set, target, temperature):
    llm.temperature = temperature
    inference_query_executor = (inference_query_prompt|llm|StrOutputParser())
    query = inference_query_executor.invoke({'input': claims, 'key_set': key_set, 'target': target})
    return query

def create_inference_query(claim_sets):
    key_words = get_key_words(claim_sets)
    for i in range(2, len(claim_sets)):
        targets = []
        claims = []
        for j in range(len(claim_sets[i])):
            targets.append(claim_sets[i][j].get("claim target"))
            claims.append(claim_sets[i][j].get("claim"))
        
        print(key_words)
        print(targets)
        print(claims)
        
        temperature = 0.01
        output = generate_inference_query(claim_sets[i], key_words, targets, temperature)
        return output


# evidence_list = evidence_search(pdf="Dave Grohl-Wikipedia.pdf")
# raw_claim_list = create_data_list(evidence_list=evidence_list)
# verified_claim_list = create_verified_list(raw_data_list=raw_claim_list) skip this step for testing to save time

# claim_sets = create_claim_sets(verified_data_list=raw_claim_list)

claim_sets = [
    [{'evidence': "sister, Lisa, three years older, was getting seriously into new wave territory. We'd meet in the middle sometimes with Bowie and Siouxsie and the Banshees.", 'claim': 'The sisters were into new wave music, specifically Bowie and Siouxsie and the Banshees.', 'claim target': 'The sisters', 'claim topic': 'New wave music'}, {'evidence': 'Grohl spent his teenage years at the club and saw some shows that changed his life.', 'claim': 'Teenage years of Grohl were significantly influenced by his experiences at the club.', 'claim target': 'Grohl', 'claim topic': "Influence of club experiences on Grohl's teenage years"}], 
    [{'evidence': 'Dave Grohl is an American musician. He is the founder of the rock band Foo Fighters, for which he is the lead singer, guitarist, and principal songwriter. Prior to forming Foo Fighters, he was the drummer of the grunge rock band Nirvana from 1990 to 1994.', 'claim': 'Dave Grohl is the founder and frontman of the rock band Foo Fighters, having previously been the drummer for Nirvana.', 'claim target': 'Dave Grohl', 'claim topic': 'Musician and founder of Foo Fighters'}, {'evidence': "released his debut documentary, Sound City, in 2013. It was followed by the documentary miniseries Sonic Highways (2014) and the documentary film What Drives Us (2021). In 2021, he and the Foo Fighters starred as themselves in the comedy horror film Studio 666. In 2010, Grohl was described by the Classic Rock Drummers co-author Ken Micallef as one of the most influential rock musicians of the previous 20 years. Grohl was inducted into the Rock and Roll Hall of Fame as part of Nirvana in 2014 and as a member of Foo Fighters in 2021. Grohl was born in Warren, Ohio, on January 14, 1969, the son of teacher Virginia Jean (née Hanlon) and newswriter James Harper Grohl. He is of German, Slovak (on father's side), Irish and English (on mother's side) descent.", 'claim': 'Dave Grohl, an influential rock musician, has had a successful career with various projects, including documentaries, films, and music bands. He was born in Warren, Ohio, and is of German, Slovak, Irish, and English descent.', 'claim target': 'Dave Grohl', 'claim topic': 'Career and descent'}, {'evidence': "Grohl became the drummer for Nirvana after Scream broke up in 1990. He has also formed Foo Fighters as a one-man project and has released 11 studio albums with them. Grohl is also the drummer and co-founder of Them Crooked Vultures and has recorded and toured with Queens of the Stone Age and Tenacious D. He has participated in the side projects Late! and Probot. Grohl began directing Foo Fighters music videos in 1997 and released his debut documentary, Sound City, in 2013. It was followed by the documentary miniseries Sonic Highways (2014) and the documentary film What Drives Us (2021). In 2022, Grohl released his memoir, 'The Storyteller: Tales of Life and Music.'", 'claim': 'Dave Grohl has been a successful musician, having formed and led various bands, including Nirvana, Foo Fighters, Them Crooked Vultures, and having released 11 studio albums, directed music videos, and written a memoir.', 'claim target': 'Dave Grohl', 'claim topic': 'Musical career and achievements'}], 
    [{'evidence': "recording a live album (their show of May 4, 1990, in Alzey, Germany, being released by Tobby Holzinger as Your Choice Live Series Vol.10) and two studio albums, No More Censorship and Fumble, on which Grohl penned and sang vocals on the song 'Gods Look Down'. During a Toronto stop on their 1987 tour, Grohl played drums for Iggy Pop at a CD release party held at famed club the El Mocambo. [24] In 1990, Scream unexpectedly disbanded midtour following the departure of bassist Skeeter Thompson. [25] While playing in Scream, Grohl became a fan of the Melvins and eventually befriended them. [26] During a 1990 tour stop on the West Coast, Melvins' guitarist Buzz Osborne took his friends Kurt Cobain and Krist Novoselic, both then with Nirvana, to see a Scream performance. [27] Following the breakup of Scream, Grohl called Osborne for advice. [28] Osborne informed him that Nirvana was seeking a drummer, and gave Grohl the phone numbers of Cobain and Novoselic, who invited him to audition. Grohl joined Nirvana in 1990, and the band's fortunes changed dramatically.", 'claim': "Grohl's friendship with the Melvins led him to join Nirvana in 1990, significantly impacting the band's success.", 'claim target': 'Grohl, Melvins, Nirvana', 'claim topic': "Impact of friendship on band's success"}, {'evidence': "Grohl's initial months in Olympia, Washington, saw him working on a song called 'Color Pictures of a Marigold'. Cobain overheard him and they worked on it together. Grohl later recorded the song for the Pocketwatch cassette. Nirvana later re-recorded the song as a B-side on the 'Heart-Shaped Box' single, titled 'Marigold'. Grohl also contributed the main guitar riff for 'Scentless Apprentice'. Cobain initially thought the riff was 'kind of boneheaded', but was pleased with the final outcome.", 'claim': "During the initial stages of his musical career in Olympia, Washington, Dave Grohl collaborated with Kurt Cobain on the song 'Color Pictures of a Marigold'. Later, Nirvana re-recorded the song as a B-side and released it as 'Marigold'. Grohl also contributed the main guitar riff for 'Scentless Apprentice'. Cobain initially had reservations about the riff, but was satisfied with the final product.", 'claim target': 'Dave Grohl and Kurt Cobain', 'claim topic': 'Musical collaboration and songwriting'}], 
    [{'evidence': "Cobain was absent during most of Nirvana's 1994 European tour session time at Robert Lang Studios in Seattle. Novoselic and Grohl worked on demos of their own songs, completing several of Grohl's future Foo Fighters songs. Cobain arrived on the third day and recorded a demo of 'You Know You're Right'. Nirvana's final studio recording. Cobain was found dead of a self-inflicted shotgun wound at his home in 1994. Grohl was inducted into the Rock and Roll Hall of Fame as a member of Nirvana in 2014 and paid tribute to Cobain by performing 'Smells Like Teen Spirit' and 'Lithium' with surviving members.", 'claim': "During Nirvana's 1994 European tour session time, Novoselic and Grohl worked on demos of their own songs, completing several of Grohl's future Foo Fighters songs.", 'claim target': 'Novoselic and Grohl', 'claim topic': 'Working on demos for future Foo Fighters songs'}, {'evidence': "Nirvana's final studio recording was made in April 1994, where Kurt Cobain was found dead of a self-inflicted shotgun wound. Grohl went into isolation and retreated to County Kerry, Ireland, in 1994. He formed the Foo Fighters as a one-man project, recording all instruments and vocals himself.", 'claim': "After Nirvana's final studio recording in 1994, Grohl formed the Foo Fighters as a one-man project in isolation in County Kerry, Ireland.", 'claim target': 'Grohl, Dave', 'claim topic': 'Formation of Foo Fighters'}]
]

# key_words = get_key_words(claim_sets=claim_sets) # Ret: [["", ""], ["", "", ""], ["", "", ""]]
# print(key_words)

print(create_inference_query(claim_sets=claim_sets))

# passing multiple parameters: {'input': blah, 'key_set': blah, 'target': blah}

# FOR LATER
# database of examples w a variety of domains to pull relevant examples to claim topic (ex. claim topic = sports, then examples relevant to sports)
