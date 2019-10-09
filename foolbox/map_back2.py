
# Timeline-based feature-to-raw-data mapback

import json
import os
import random
import sys


def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def write_json(file_to_write, content_to_write):
    with open(file_to_write, 'w') as outfile:
        json.dump(content_to_write, outfile, indent=2)

def generate_node_creation_event(node_id,
                                 actor_id,
                                 node_type,
                                 tag_name):
    creation_event = {}
    creation_event["event_type"] = "NodeCreation"
    creation_event["node_type"] = int(node_type)
    creation_event["tag_name"] = str(tag_name)
    creation_event["node_id"] = str(node_id)
    creation_event["actor_id"] = str(actor_id)
    return creation_event


def generate_node_insertion_event(node_id,
                                  actor_id,
                                  parent_node_id,
                                  previous_sibling_node_id,
                                  tag_name,
                                  node_type,
                                  parent_node_type,
                                  node_attributes=[]):
    insertion_event = {}
    insertion_event["node_previous_sibling_id"] = str(previous_sibling_node_id)
    insertion_event["event_type"] = "NodeInsertion"
    insertion_event["parent_node_type"] = int(parent_node_type)
    insertion_event["node_parent_id"] = str(parent_node_id)
    insertion_event["node_type"] = int(node_type)
    insertion_event["tag_name"] = str(tag_name)
    insertion_event["node_id"] = str(node_id)
    insertion_event["actor_id"] = str(actor_id)
    insertion_event["node_attributes"] = node_attributes
    return insertion_event


def generate_node_removal_event(node_id, actor_id):
    removal_event = {}
    removal_event["node_id"] = str(node_id)
    removal_event["actor_id"] = str(actor_id)
    removal_event["event_type"] = "NodeRemoval"
    return removal_event


def generate_script_compilation_event(node_id,
                                      script_text,
                                      script_id,
                                      script_url):
    script_compilation_event = {}
    script_compilation_event["node_id"] = str(node_id)
    script_compilation_event["script_id"] = str(script_id)
    script_compilation_event["script_text"] = str(script_text)
    script_compilation_event["script_url"] = str(script_url)
    script_compilation_event["event_type"] = "ScriptCompilation"
    return script_compilation_event


def generate_script_execution_event(node_id,
                                    script_text,
                                    script_id,
                                    script_url,
                                    in_execution):
    script_execution_event = {}
    script_execution_event["node_id"] = str(node_id)
    script_execution_event["script_id"] = str(script_id)
    script_execution_event["script_text"] = str(script_text)
    script_execution_event["script_url"] = str(script_url)
    script_execution_event["event_type"] = "ScriptExecution"
    script_execution_event["in_execution"] = bool(in_execution)
    return script_execution_event



####################
# Mapback methods #
####################

# Feature: FEATURE_GRAPH_NODES
# Mapback #1: Insert additional nodes 

def insert_nodes(timeline, delta, request_url=None):
    if delta <= 0:
        return
    
    max_node_id = -1
    max_script_id = -1

    request_node_previous_sibling_id = '0'
    request_node_id = '0'

    new_timeline = []

    # get variables
    for event in timeline['timeline']:
        if "node_id" in event:
            if int(event['node_id']) > max_node_id:
                max_node_id = int(event['node_id'])
        if "script_id" in event:
            if int(event['script_id']) > max_script_id:
                max_script_id = int(event['script_id'])
        if "node_attributes" in event:
            for attr in event["node_attributes"]:
                if attr["attr_value"] == request_url:
                    request_node_id = event["node_id"]
                    request_node_parent_id = event["node_parent_id"]
                    request_node_parent_type = event["parent_node_type"]
                    request_node_previous_sibling_id = event["node_previous_sibling_id"]

    dummy_node_ids = []
    curr_node_id = max_node_id + 1
    curr_script_id = max_script_id + 1
    curr_node_previous_sibling_id = int(request_node_previous_sibling_id)

    for event in timeline['timeline']:
        if "node_id" in event:
            if event['node_id'] == request_node_id and event["event_type"] == "NodeCreation":
                # step 1: create and insert script compilation/execution node
                script_node_creation_event = generate_node_creation_event(
                    node_id=curr_node_id,
                    actor_id="0",
                    tag_name="script",
                    node_type=3
                )
                script_node_insertion_event = generate_node_insertion_event(
                    node_id=curr_node_id,
                    actor_id="0",
                    node_type=3,
                    tag_name="",
                    parent_node_id=request_node_parent_id,
                    previous_sibling_node_id=curr_node_previous_sibling_id,
                    parent_node_type=request_node_parent_type
                )
                new_timeline.append(
                    script_node_creation_event)
                new_timeline.append(
                    script_node_insertion_event)

                curr_node_previous_sibling_id += 1
                script_compilation_event = generate_script_compilation_event(
                    node_id=curr_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url"
                )
                script_execution_start_node_id = curr_node_id
                script_execution_start_event = generate_script_execution_event(
                    node_id=script_execution_start_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url",
                    in_execution=True
                )
                new_timeline.append(
                    script_compilation_event)
                new_timeline.append(
                    script_execution_start_event)
                curr_node_id += 1

                # step 2: create and insert dummy nodes
                for i in range(delta):
                    curr_node_previous_sibling_id = curr_node_id - 1
                    dummy_node_creation_event = generate_node_creation_event(
                        node_id=curr_node_id,
                        actor_id=curr_script_id,
                        tag_name="div",
                        node_type=1
                    )
                    dummy_node_insertion_event = generate_node_insertion_event(
                        node_id=curr_node_id,
                        actor_id=curr_script_id,
                        parent_node_id=request_node_parent_id,
                        previous_sibling_node_id=curr_node_previous_sibling_id,
                        tag_name="DIV",
                        node_type=1,
                        parent_node_type=1
                    )
                    new_timeline.append(
                        dummy_node_creation_event)
                    new_timeline.append(
                        dummy_node_insertion_event)
                    dummy_node_ids.append(curr_node_id)
                    curr_node_id += 1

                # step 3: create and insert script execution end event
                script_execution_end_event = generate_script_execution_event(
                    node_id=script_execution_start_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url",
                    in_execution=False
                )
                new_timeline.append(
                    script_execution_end_event)

        # step 4: modify the previous_subling_id of the NodeInsertion event of target request
        # to make sure that it now points to the last dummy node
        if "node_attributes" in event:
            for attr in event["node_attributes"]:
                if attr["attr_value"] == request_url:
                    event["node_previous_sibling_id"] = str(curr_node_id)

        # step 5: lastly add node removal events to remove previously added dummy nodes
        if "request_url" in event and "requestor_id" in event:
            if event["request_url"] == request_url:
                new_timeline.append(event)
                for i in range(len(dummy_node_ids)):
                    removal_event = generate_node_removal_event(
                        node_id=dummy_node_ids[i],
                        actor_id=curr_script_id
                    )
                    new_timeline.append(removal_event)
                continue

        new_timeline.append(event)

    timeline['timeline'] = new_timeline


def insert_random_global_nodes(timeline, delta, request_url=None):
    if delta <= 0:
        return
    
    max_node_id = -1
    max_script_id = -1

    request_node_previous_sibling_id = '0'
    request_node_id = '0'

    # get variables
    for event in timeline['timeline']:
        if "node_id" in event:
            if int(event['node_id']) > max_node_id:
                max_node_id = int(event['node_id'])
        if "script_id" in event:
            if int(event['script_id']) > max_script_id:
                max_script_id = int(event['script_id'])
        if "node_attributes" in event:
            for attr in event["node_attributes"]:
                if attr["attr_value"] == request_url:
                    request_node_id = event["node_id"]
                    request_node_parent_id = event["node_parent_id"]
                    request_node_parent_type = event["parent_node_type"]
                    request_node_previous_sibling_id = event["node_previous_sibling_id"]

    dummy_node_ids = []
    curr_node_id = max_node_id + 1
    curr_script_id = max_script_id + 1

    new_timeline = []

    nodes_seen = set()
    first_node = True

    for event in timeline['timeline']:
        if 'node_id' in event:
            if event['event_type'] == 'NodeInsertion':
                if not ('tag_name' in event and event['tag_name'].lower() == 'script'):
                    if first_node:
                        nodes_seen.add(int(event['node_id']))
                        if 'node_parent_id' in event:
                            nodes_seen.add(int(event['node_parent_id']))
                        first_node = False
                    else:
                        if 'actor_id' in event and event['actor_id'] == '0' and 'node_parent_id' in event and int(event['node_parent_id']) in nodes_seen:
                            nodes_seen.add(int(event['node_id']))

            if event['node_id'] == request_node_id and event['event_type'] == 'NodeCreation':
                # step 1: create and insert script compilation/execution node
                script_node_creation_event = generate_node_creation_event(
                    node_id=curr_node_id,
                    actor_id="0",
                    tag_name="script",
                    node_type=3
                )
                script_node_insertion_event = generate_node_insertion_event(
                    node_id=curr_node_id,
                    actor_id="0",
                    node_type=3,
                    tag_name="",
                    parent_node_id=request_node_parent_id,
                    previous_sibling_node_id=request_node_previous_sibling_id,
                    parent_node_type=request_node_parent_type
                )
                new_timeline.append(
                    script_node_creation_event)
                new_timeline.append(
                    script_node_insertion_event)

                script_compilation_event = generate_script_compilation_event(
                    node_id=curr_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url"
                )
                script_execution_start_node_id = curr_node_id
                script_execution_start_event = generate_script_execution_event(
                    node_id=script_execution_start_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url",
                    in_execution=True
                )
                new_timeline.append(
                    script_compilation_event)
                new_timeline.append(
                    script_execution_start_event)
                curr_node_id += 1

                # step 2: create and insert dummy nodes
                nodes_seen_list = list(nodes_seen)
                for i in range(delta):
                    dummy_node_creation_event = generate_node_creation_event(
                        node_id=curr_node_id,
                        actor_id=0,
                        tag_name="div",
                        node_type=1
                    )
                    dummy_node_insertion_event = generate_node_insertion_event(
                        node_id=curr_node_id,
                        actor_id=0,
                        parent_node_id=random.choice(nodes_seen_list),
                        previous_sibling_node_id=0,#random.choice(nodes_seen_list),#curr_node_previous_sibling_id,
                        tag_name="DIV",
                        node_type=1,
                        parent_node_type=1
                    )
                    new_timeline.append(
                        dummy_node_creation_event)
                    new_timeline.append(
                        dummy_node_insertion_event)
                    dummy_node_ids.append(curr_node_id)
                    curr_node_id += 1

                # step 3: create and insert script execution end event
                script_execution_end_event = generate_script_execution_event(
                    node_id=script_execution_start_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url",
                    in_execution=False
                )
                new_timeline.append(
                    script_execution_end_event)

        if "request_url" in event and "requestor_id" in event:
            if event["request_url"] == request_url:
                new_timeline.append(event)
                for i in range(len(dummy_node_ids)):
                    removal_event = generate_node_removal_event(
                        node_id=dummy_node_ids[i],
                        actor_id=curr_script_id
                    )
                    new_timeline.append(removal_event)
                continue

        new_timeline.append(event)
    
    timeline['timeline'] = new_timeline




# Feature: FEATURE_GRAPH_EDGES

def insert_random_edges(timeline, delta, request_url=None):
    if delta <= 0:
        return
    
    max_node_id = -1
    max_script_id = -1

    request_node_previous_sibling_id = '0'
    request_node_id = '0'

    # get variables
    for event in timeline['timeline']:
        if "node_id" in event:
            if int(event['node_id']) > max_node_id:
                max_node_id = int(event['node_id'])
        if "script_id" in event:
            if int(event['script_id']) > max_script_id:
                max_script_id = int(event['script_id'])
        if "node_attributes" in event:
            for attr in event["node_attributes"]:
                if attr["attr_value"] == request_url:
                    request_node_id = event["node_id"]
                    request_node_parent_id = event["node_parent_id"]
                    request_node_parent_type = event["parent_node_type"]
                    request_node_previous_sibling_id = event["node_previous_sibling_id"]

    dummy_node_ids = []
    curr_node_id = max_node_id + 1
    curr_script_id = max_script_id + 1

    new_timeline = []

    nodes_seen = set()
    first_node = True

    for event in timeline['timeline']:
        if 'node_id' in event:
            if event['event_type'] == 'NodeInsertion':
                if not ('tag_name' in event and event['tag_name'].lower() == 'script'):
                    if first_node:
                        nodes_seen.add(int(event['node_id']))
                        if 'node_parent_id' in event:
                            nodes_seen.add(int(event['node_parent_id']))
                        first_node = False
                    else:
                        if 'actor_id' in event and event['actor_id'] == '0' and 'node_parent_id' in event and int(event['node_parent_id']) in nodes_seen:
                            nodes_seen.add(int(event['node_id']))

            if event['node_id'] == request_node_id and event['event_type'] == 'NodeCreation':
                # step 1: create and insert script compilation/execution node
                script_node_creation_event = generate_node_creation_event(
                    node_id=curr_node_id,
                    actor_id="0",
                    tag_name="script",
                    node_type=3
                )
                script_node_insertion_event = generate_node_insertion_event(
                    node_id=curr_node_id,
                    actor_id="0",
                    node_type=3,
                    tag_name="",
                    parent_node_id=request_node_parent_id,
                    previous_sibling_node_id=request_node_previous_sibling_id,
                    parent_node_type=request_node_parent_type
                )
                new_timeline.append(
                    script_node_creation_event)
                new_timeline.append(
                    script_node_insertion_event)

                script_compilation_event = generate_script_compilation_event(
                    node_id=curr_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url"
                )
                script_execution_start_node_id = curr_node_id
                script_execution_start_event = generate_script_execution_event(
                    node_id=script_execution_start_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url",
                    in_execution=True
                )
                new_timeline.append(
                    script_compilation_event)
                new_timeline.append(
                    script_execution_start_event)
                curr_node_id += 1

                # step 2: create and insert dummy nodes
                nodes_seen_list = list(nodes_seen)
                for i in range(delta):
                    dummy_node_creation_event = generate_node_creation_event(
                        node_id=curr_node_id,
                        actor_id=0,
                        tag_name="div",
                        node_type=1
                    )
                    dummy_node_insertion_event = generate_node_insertion_event(
                        node_id=curr_node_id,
                        actor_id=0,
                        parent_node_id=random.choice(nodes_seen_list),
                        previous_sibling_node_id=0,#random.choice(nodes_seen_list),#curr_node_previous_sibling_id,
                        tag_name="DIV",
                        node_type=1,
                        parent_node_type=1
                    )
                    new_timeline.append(
                        dummy_node_creation_event)
                    new_timeline.append(
                        dummy_node_insertion_event)
                    dummy_node_ids.append(curr_node_id)
                    curr_node_id += 1

                # step 3: create and insert script execution end event
                script_execution_end_event = generate_script_execution_event(
                    node_id=script_execution_start_node_id,
                    script_id=curr_script_id,
                    script_text="dummy",
                    script_url="dummy_url",
                    in_execution=False
                )
                new_timeline.append(
                    script_execution_end_event)

        if "request_url" in event and "requestor_id" in event:
            if event["request_url"] == request_url:
                new_timeline.append(event)
                for i in range(len(dummy_node_ids)):
                    removal_event = generate_node_removal_event(
                        node_id=dummy_node_ids[i],
                        actor_id=curr_script_id
                    )
                    new_timeline.append(removal_event)
                continue

        new_timeline.append(event)
    
    timeline['timeline'] = new_timeline


# Feature: FEATURE_GRAPH_NODES_EDGES

# Feature: FEATURE_GRAPH_EDGES_NODES




# feature mapback methods
feature_mapbacks = {
    "FEATURE_GRAPH_NODES": [insert_nodes, insert_random_global_nodes],
    "FEATURE_GRAPH_EDGES": [insert_random_edges],
}



def mapback(timeline, feature, delta, request_url=None):
    # always use the first mapback method
    choice = 1
    feature_mapbacks[feature][choice](timeline, delta, request_url)


if __name__ == "__main__":
    timeline_file = sys.argv[1]
    timeline = read_json(timeline_file)
    request_url = "https://www.gstatic.com/og/_/js/k=og.og2.en_US.GdU2tvYwRSE.O/rt=j/m=def,aswid/exm=in,fot/d=1/ed=1/rs=AA2YrTvGmug-unAddkGu09wbESqi62SHzg"
    #mapback(timeline, "FEATURE_GRAPH_NODES", 1000, request_url)
    mapback(timeline, "FEATURE_GRAPH_EDGES", 1000, request_url)
    write_json(timeline_file + ".modified", timeline)


