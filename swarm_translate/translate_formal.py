import json

input_source_sentences = []

with open('/home/clear/swarm/swarm_translate/scenarios/data/vref.txt') as id_file, \
     open('/home/clear/swarm/swarm_translate/scenarios/data/eng-engylt.txt') as content_file:
    
    for id_line, content_line in zip(id_file, content_file):
        entry = {
            "id": id_line.strip(),
            "content": content_line.strip()
        }
        input_source_sentences.append(entry)
