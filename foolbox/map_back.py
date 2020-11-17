import os
import json
import numpy as np
import random

# THIS MODULE MAPS PERTURBATIONS FROM FEATURE-SPACE TO TIMELINE-SPACE (WE CONSIDER IT
# AS A EQUVALENCE OF WEBPAGE-SPACE)

HOME_DIR = os.getenv("HOME")
BASE_TIMELINE_DIR = HOME_DIR + '/rendering_stream/'
BASE_DEF_DIR = HOME_DIR + "/attack-adgraph-pipeline/def/"

TEST_MODE = 2

CATEGORICAL_FEATURE_IDX = {
    "FEATURE_NODE_CATEGORY",
    "FEATURE_FIRST_PARENT_TAG_NAME",
    "FEATURE_FIRST_PARENT_SIBLING_TAG_NAME",
    "FEATURE_SECOND_PARENT_TAG_NAME",
    "FEATURE_SECOND_PARENT_SIBLING_TAG_NAME"
}

UNUSED_FEATURE_IDX = {
    "DOMAIN_NAME",
    "NODE_ID",
    "FEATURE_KATZ_CENTRALITY",
    "FEATURE_FIRST_PARENT_KATZ_CENTRALITY",
    "FEATURE_SECOND_PARENT_KATZ_CENTRALITY"
}


def write_json(file_to_write, content_to_write):
    with open(file_to_write, 'w') as outfile:
        json.dump(content_to_write, outfile, indent=2)


def write_html(file_to_write, content_to_write):
    if isinstance(content_to_write, str):
        with open(file_to_write, 'w') as outfile:
            outfile.write(content_to_write)
    else:
        with open(file_to_write, 'wb') as outfile:
            html_dump = content_to_write.prettify().encode("utf-8")
            outfile.write(html_dump)


def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_largest_fname(path, filter_keyword=[".modified", "parsed_"]):
    largest_size = -1
    largest_fname, largest_fpath = "", ""
    for fname in os.listdir(path):
        should_fliter = False
        for kw in filter_keyword:
            if kw in fname:
                should_fliter = True
                break
        if should_fliter:
            continue
        fpath = os.path.join(path, fname)
        if os.path.getsize(fpath) > largest_size:
            largest_size = os.path.getsize(fpath)
            largest_fname = fname
            largest_fpath = fpath
    # input(largest_fname + largest_fpath)
    return largest_fname, largest_fpath


# def get_file_names(domain):
#     domain_data_dir = BASE_TIMELINE_DIR + "data/" + domain
#     original_json_fname, original_json_fpath = get_largest_fname(
#         domain_data_dir)
#     original_json = read_json(original_json_fpath)
#     url_id_map_fname = BASE_TIMELINE_DIR + "features/" + domain + '.csv'
#     if not os.path.isfile(url_id_map_fname):
#         generate_url_id_map_file(domain)
#     url_id_map = read_features(url_id_map_fname)
#     return original_json, original_json_fpath, url_id_map


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


def modify_json(json, domain, request_id, diff, url_id_map, original_json_fname=""):
    rendering_stream = json

    rendering_stream_url = rendering_stream['url']
    max_node_id = -1
    max_script_id = -1
    request_url = url_id_map[request_id]

    # This new JSON has (simulated) perturbations that are added nodes
    updated_rendering_stream = {}
    updated_rendering_stream['url'] = rendering_stream_url
    updated_rendering_stream['timeline'] = []

    request_node_previous_sibling_id = '0'
    request_node_parent_id = -1
    request_node_id = '0'
    request_node_parent_type = 1

    # pass 1
    for event in rendering_stream['timeline']:
        if "node_id" in event:
            if int(event['node_id']) > max_node_id:
                max_node_id = int(event['node_id'])
        if "script_id" in event:
            if int(event['script_id']) > max_script_id:
                max_script_id = int(event['script_id'])
        if "node_attributes" in event:
            for attr in event["node_attributes"]:
                if attr["attr_value"] == request_url:
                    request_node_parent_id = event["node_parent_id"]
                    request_node_id = event["node_id"]
                    request_node_parent_type = event["parent_node_type"]
                    request_node_previous_sibling_id = event["node_previous_sibling_id"]
        if "request_url" in event:
            if event['request_url'] == request_url:
                # input("Found URL!")
                request_actor_id = event['actor_id']
                request_node_id = event["requestor_id"]
        if "tag_name" in event and "event_type" in event:
            if event["tag_name"] == "BODY" and request_node_parent_id == -1:
                request_node_parent_id = event["node_id"]

    # pass 2
    dummy_node_ids = []
    curr_node_id = max_node_id + 1
    curr_script_id = max_script_id + 1
    curr_node_previous_sibling_id = int(request_node_previous_sibling_id)

    # input(request_node_id)

    for event in rendering_stream['timeline']:
        updated_event = event
        if 0 in diff:
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
                    updated_rendering_stream['timeline'].append(
                        script_node_creation_event)
                    updated_rendering_stream['timeline'].append(
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
                    updated_rendering_stream['timeline'].append(
                        script_compilation_event)
                    updated_rendering_stream['timeline'].append(
                        script_execution_start_event)
                    curr_node_id += 1

                    # print("Before insersion:" + str(len(updated_rendering_stream['timeline'])))
                    # step 2: create and insert dummy nodes
                    for i in range(diff[0]):
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
                        updated_rendering_stream['timeline'].append(
                            dummy_node_creation_event)
                        updated_rendering_stream['timeline'].append(
                            dummy_node_insertion_event)
                        dummy_node_ids.append(curr_node_id)
                        curr_node_id += 1
                    # print("After insersion:" + str(len(updated_rendering_stream['timeline'])))

                    # step 3: create and insert script execution end event
                    script_execution_end_event = generate_script_execution_event(
                        node_id=script_execution_start_node_id,
                        script_id=curr_script_id,
                        script_text="dummy",
                        script_url="dummy_url",
                        in_execution=False
                    )
                    updated_rendering_stream['timeline'].append(
                        script_execution_end_event)

            # step 4: modify the previous_subling_id of the NodeInsertion event of target request
            # to make sure that it now points to the last dummy node
            if "node_attributes" in event:
                for attr in event["node_attributes"]:
                    if attr["attr_value"] == request_url:
                        updated_event["node_previous_sibling_id"] = str(
                            curr_node_id)

        # step 5: lastly add node removal events to remove previously added dummy nodes
        if "request_url" in event and "requestor_id" in event:
            if event["request_url"] == request_url:
                # sub-step 1: perturbe length of URL
                q_str = "?q="
                perturbed_url = request_url
                if diff[328] > 3:
                    # input("Now perturbing URL length...")
                    perturbed_url = request_url + q_str + (diff[328] - 3) * 'a'
                elif diff[328] > 0:
                    perturbed_url = request_url + q_str[:diff[328]]
                # input(str(event) + str(perturbed_url))
                updated_event["request_url"] = perturbed_url
                updated_rendering_stream['timeline'].append(updated_event)

                # sub-step 2: handle removal nodes
                if 0 in diff:
                    for i in range(len(dummy_node_ids)):
                        removal_event = generate_node_removal_event(
                            node_id=dummy_node_ids[i],
                            actor_id=curr_script_id
                        )
                        updated_rendering_stream['timeline'].append(removal_event)
                    continue

        updated_rendering_stream['timeline'].append(updated_event)

    return updated_rendering_stream


def read_features(fpath):
    features = {}
    with open(fpath, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        url_id = r.split(",")[1]
        features[url_id] = r
    return features


def read_feature_def(unencoded_idx_fpath, encoded_idx_fpath):
    unencoded_feature_def = {}
    with open(unencoded_idx_fpath, 'r') as fin:
        unencoded_data = fin.readlines()
    for r in unencoded_data:
        r = r.strip()
        idx, feature_name = r.split(",")
        unencoded_feature_def[int(idx)] = feature_name

    encoded_feature_def = {}
    idx_to_feature_name_map = {}
    for f in list(CATEGORICAL_FEATURE_IDX):
        encoded_feature_def[f] = {}
    with open(encoded_idx_fpath, 'r') as fin:
        encoded_data = fin.readlines()
    for r in encoded_data:
        r = r.strip()
        idx, feature_name = r.split(",")
        if '=' in feature_name:
            # "1" prevents spliting on tag name itself
            name, val = feature_name.split("=", 1)
            if name in CATEGORICAL_FEATURE_IDX:
                encoded_feature_def[name][val] = len(encoded_feature_def[name])
            idx_to_feature_name_map[int(idx)] = name
        else:
            idx_to_feature_name_map[int(idx)] = feature_name

    return unencoded_feature_def, encoded_feature_def, idx_to_feature_name_map


def one_hot_encode_x(x, unencoded_feature_def, encoded_feature_def):
    encoded_x = []
    for i in range(len(x)):
        feature_name = unencoded_feature_def[i]
        if feature_name in UNUSED_FEATURE_IDX:
            continue
        if i == len(x) - 1:
            continue
        if feature_name in CATEGORICAL_FEATURE_IDX:
            # handle missing value
            if x[i] == '':
                # replace the missing value with a default value
                # x[i] = list(encoded_feature_def[feature_name].keys())[0]
                x[i] = "?"
            for j in range(len(encoded_feature_def[feature_name])):
                if j == encoded_feature_def[feature_name][x[i]]:
                    encoded_x.append(float(1.0))
                else:
                    encoded_x.append(float(0.0))
        else:
            encoded_x.append(x[i])
    return encoded_x


def read_feature_stats(fname):
    feature_stats = {}
    with open(fname, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        feature_name, maxn, minn = r.split(",")
        feature_stats[feature_name] = [maxn, minn]
    return feature_stats


def normalize_x(x, encoded_feature_def, feature_stats, idx_to_feature_name_map):
    def scale_feature(val, stats):
        [maxn, minn] = stats
        val = float(val)
        maxn, minn = float(maxn), float(minn)
        if maxn == minn:
            return val
        return (val - minn) / (maxn - minn)

    normalized_x = []

    for i in range(len(x)):
        val = x[i]
        feature_name = idx_to_feature_name_map[i]
        if feature_name in feature_stats:
            stats = feature_stats[feature_name]
            scaled_val = scale_feature(val, stats)
            normalized_x.append(scaled_val)
        else:
            normalized_x.append(val)

    return normalized_x


def get_x_from_features(domain, target_url_id):
    features_fpath = BASE_TIMELINE_DIR + "features/" + domain + '.csv'
    features = read_features(features_fpath)
    x_str = features[target_url_id]
    x_lst = x_str.split(",")
    return x_lst


def run_cpp_feature_extractor(domain_url, working_dir, parse_modified):
    def execute_shell_command(cmd):
        os.system(cmd)

    cmd_lst = []
    cmd_lst.append("cd %s" % working_dir)
    if parse_modified:
        cmd_lst.append("sh test.sh %s" % domain_url)
    else:
        cmd_lst.append("sh test.sh %s parse-unmod" % domain_url)
    cmd = ' && '.join(cmd_lst)
    print("Issuing shell command: " + cmd)
    execute_shell_command(cmd)


def compute_x_after_mapping_back(domain_url, url_id, modified_html, original_html_fname, working_dir):
    write_html(BASE_TIMELINE_DIR + 'html/' + 'modified_' + original_html_fname, modified_html)
    run_cpp_feature_extractor(domain_url, working_dir, parse_modified=True)
    new_x = get_x_from_features(domain_url, url_id)

    unencoded_feature_def, encoded_feature_def, idx_to_feature_name_map = read_feature_def(
        BASE_DEF_DIR + "unnormalized_feature_idx.csv",
        BASE_DEF_DIR + "trimmed_wo_class_feature_idx.csv"
    )
    one_hot_encoded_x = one_hot_encode_x(
        new_x, unencoded_feature_def, encoded_feature_def)

    feature_stats = read_feature_stats(
        BASE_DEF_DIR + "col_stats_for_unnormalization.csv")
    normalized_x = normalize_x(
        one_hot_encoded_x, encoded_feature_def, feature_stats, idx_to_feature_name_map)
    return np.array(normalized_x).astype(np.float), new_x


####################
# Mapback methods #
####################

# Feature: FEATURE_GRAPH_NODES
# Mapback #1: Insert additional nodes


def insert_nodes(timeline, delta, request_url=None):
    if delta <= 0:
        return timeline

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
    return timeline


def insert_random_global_nodes(timeline, delta, request_url=None):
    if delta <= 0:
        return timeline

    max_node_id = -1
    max_script_id = -1
    request_node_parent_id = -1

    request_node_previous_sibling_id = '0'
    request_node_id = '0'
    request_node_parent_type = 1

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
        if "request_url" in event:
            if event['request_url'] == request_url:
                request_node_id = event["requestor_id"]
        if "tag_name" in event and "event_type" in event:
            if event["tag_name"] == "BODY" and request_node_parent_id == -1:
                request_node_parent_id = event["node_id"]

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
                        previous_sibling_node_id=0,  # random.choice(nodes_seen_list),#curr_node_previous_sibling_id,
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
    return timeline

# Feature: FEATURE_GRAPH_EDGES


def insert_random_edges(timeline, delta, request_url=None):
    if delta <= 0:
        return timeline

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
                        previous_sibling_node_id=0,  # random.choice(nodes_seen_list),#curr_node_previous_sibling_id,
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
    return timeline

# Feature: FEATURE_GRAPH_NODES_EDGES

# Feature: FEATURE_GRAPH_EDGES_NODES

# Feature: FEATURE_GRAPH_EDGES


def increase_url_length(timeline, delta, request_url=None):
    if delta <= 0:
        return timeline

    updated_rendering_stream = {}
    rendering_stream_url = timeline['url']
    updated_rendering_stream['url'] = rendering_stream_url
    updated_rendering_stream['timeline'] = []

    for event in timeline['timeline']:
        updated_event = event
        if "request_url" in event and "requestor_id" in event:
            if event["request_url"] == request_url:
                q_str = "?q="
                perturbed_url = request_url
                if delta > 3:
                    perturbed_url = request_url + q_str + (delta - 3) * 'a'
                elif delta > 0:
                    perturbed_url = request_url + q_str[:delta]
                updated_event["request_url"] = perturbed_url
        updated_rendering_stream['timeline'].append(updated_event)

    return updated_rendering_stream


# feature mapback methods
feature_mapbacks = {
    "FEATURE_GRAPH_NODES": [insert_nodes, insert_random_global_nodes],
    "FEATURE_GRAPH_EDGES": [insert_random_edges],
    "FEATURE_URL_LENGTH": [increase_url_length]
}


def mapback(timeline, delta, request_url=None):
    new_timeline = timeline
    if 0 in delta:
        # always use the first mapback method
        # input(delta)
        choice = 1
        if delta[0] > 0:
            new_timeline = feature_mapbacks["FEATURE_GRAPH_NODES"][1](timeline, delta[0], request_url)
    if 328 in delta:
        if delta[328] > 0:
            new_timeline = feature_mapbacks["FEATURE_URL_LENGTH"][0](new_timeline, delta[328], request_url)
    else:
        input("ERR: no perturbable feature idx found")
    return new_timeline
