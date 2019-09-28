import os
import json
import numpy as np

# THIS MODULE MAPS PERTURBATIONS FROM FEATURE-SPACE TO TIMELINE-SPACE (WE CONSIDER IT
# AS A EQUVALENCE OF WEBPAGE-SPACE)

HOME_DIR = os.getenv("HOME")
BASE_TIMELINE_DIR = HOME_DIR + "/rendering_stream/"
BASE_DEF_DIR = HOME_DIR + "/Desktop/attack-adgraph-pipeline/def/"

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


def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def read_features(fname):
    url_map = {}
    with open(fname, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        if len(r.split(',')) == 2:
            request_id, request_url = r.split(',')
        else:
            continue
        url_map[request_id] = request_url
    return url_map


def get_largest_fname(path, filter_keyword="modified"):
    largest_size = -1
    largest_fname, largest_fpath = "", ""
    for fname in os.listdir(path):
        if filter_keyword in fname:
            continue
        fpath = os.path.join(path, fname)
        if os.path.getsize(fpath) > largest_size:
            largest_size = os.path.getsize(fpath)
            largest_fname = fname
            largest_fpath = fpath
    return largest_fname, largest_fpath


def get_file_names(domain):
    domain_data_dir = BASE_TIMELINE_DIR + "data/" + domain
    original_json_fname, original_json_fpath = get_largest_fname(
        domain_data_dir)
    original_json = read_json(original_json_fpath)
    url_id_map_fname = BASE_TIMELINE_DIR + "features/" + domain + '.csv'
    if not os.path.isfile(url_id_map_fname):
        generate_url_id_map_file(domain)
    url_id_map = read_features(url_id_map_fname)
    return original_json, original_json_fpath, url_id_map


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


def modify_json(json, domain, request_id, diff, url_id_map):
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
    request_node_id = '0'

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
            if event['request_url'].strip() == request_url:
                request_actor_id = event['actor_id']
                requestor_id = event["requestor_id"]

    # pass 2
    dummy_node_ids = []
    curr_node_id = max_node_id + 1
    curr_script_id = max_script_id + 1
    curr_node_previous_sibling_id = int(request_node_previous_sibling_id)

    for event in rendering_stream['timeline']:
        updated_event = event
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
                updated_rendering_stream['timeline'].append(event)
                for i in range(len(dummy_node_ids)):
                    removal_event = generate_node_removal_event(
                        node_id=dummy_node_ids[i],
                        actor_id=curr_script_id
                    )
                    updated_rendering_stream['timeline'].append(removal_event)
                continue

        updated_rendering_stream['timeline'].append(updated_event)
    return updated_rendering_stream


def read_mapping(fpath):
    mapping = {}
    with open(fpath, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        url_id = r.split(",")[1]
        mapping[url_id] = r
    return mapping


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
                x[i] = list(encoded_feature_def[feature_name].keys())[0]
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


def get_x_from_mappings(domain, target_url_id):
    mapping_fpath = BASE_TIMELINE_DIR + "mappings/" + domain + '.csv'
    mappings = read_mapping(mapping_fpath)
    x_str = mappings[target_url_id]
    x_lst = x_str.split(",")
    return x_lst


def run_cpp_feature_extractor(domain_url, working_dir):
    def execute_shell_command(cmd):
        os.system(cmd)

    cmd_lst = []
    cmd_lst.append("cd %s" % working_dir)
    cmd_lst.append("sh test.sh %s" % domain_url)
    cmd = ' && '.join(cmd_lst)
    # print("Issuing shell command:", cmd)
    execute_shell_command(cmd)


def compute_x_after_mapping_back(domain_url, url_id, modified_json, original_json_fname, working_dir):
    write_json(original_json_fname + '.modified', modified_json)
    run_cpp_feature_extractor(domain_url, working_dir)
    new_x = get_x_from_mappings(domain_url, url_id)

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


if __name__ == "__main__":
    if TEST_MODE == 1:
        home_dir = os.getenv("HOME")
        BASE_TIMELINE_DIR = home_dir + "/rendering_stream/"
        EXAMPLE_JSON_PATH = BASE_TIMELINE_DIR + \
            "data/www.washingtonpost.com/log_www.washingtonpost.com_1568159567.916821.json"
        EXAMPLE_FEATURE_PATH = BASE_TIMELINE_DIR + "features/www.washingtonpost.com.csv"

        # For test purpose only
        rendering_stream = read_json(EXAMPLE_JSON_PATH)
        url_id_map = read_features(EXAMPLE_FEATURE_PATH)
        modified_json = modify_json(rendering_stream, "www.washingtonpost.com", "URL_24", {
                                    0: 1000, 1: 20}, url_id_map)
        write_json(EXAMPLE_JSON_PATH + '.modified', modified_json)
    elif TEST_MODE == 2:
        domain = "www.washingtonpost.com"
        url_id = "URL_27"
        diff = {0: 1000, 1: 20}
        working_dir = "~/Desktop/AdGraphAPI/scripts"
        original_json, original_json_fname, url_id_map = get_file_names(domain)
        modified_json = modify_json(
            original_json, domain, url_id, diff, url_id_map)
        mapped_x = compute_x_after_mapping_back(
            domain, url_id, modified_json, original_json_fname, working_dir)
        print(mapped_x)
