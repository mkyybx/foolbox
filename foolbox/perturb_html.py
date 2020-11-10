#!/usr/bin/env python3

import html
import re
import sys
from bs4 import BeautifulSoup

# process features
deltaNodes = 0  # # of edges should be equal to # of nodes
soup = None
requestURL = None


def BSFilterByAnyString(tag):
    # print(type(tag))
    if requestURL in tag.attrs.values():
        return tag


def addNodes(delta, mandatory=True):
    global deltaNodes, soup
    if delta < 0:
        print("changeNodes with delta < 0!")
        return
    if deltaNodes >= delta and mandatory:
        # We have done this
        return
    tag = soup.find(BSFilterByAnyString)
    for i in range(delta):
        deltaNodes += 1
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        tag.insert_after(newTag)


def wrapNodes(layers, tagName="span"):
    global deltaNodes, soup
    if layers < 0:
        print("wrapNodes with layers < 0!")
        return
    for i in range(layers):
        tag = soup.find(BSFilterByAnyString)
        deltaNodes += 1
        newTag = soup.new_tag(tagName)
        tag.wrap(newTag)


def IncreaseFirstNumberOfSiblings(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        tag.insert_after(newTag)


def DecreaseFirstNumberOfSiblings(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    orig_sibling_num = -1
    for sibling in tag.parent.children:
        orig_sibling_num += 1
    print("orig_sibling_num: %d" % orig_sibling_num)
    newTag = soup.new_tag('span')
    deltaNodes += 1
    tag.wrap(newTag)
    for i in range(orig_sibling_num - delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        tag.insert_after(newTag)


def IncreaseFirstParentNumberOfSiblings(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        parent.insert_after(newTag)


def DecreaseFirstParentNumberOfSiblings(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    orig_sibling_num = -1
    for sibling in parent.parent.children:
        orig_sibling_num += 1
    print("orig_sibling_num: %d" % orig_sibling_num)
    newTag = soup.new_tag('span')
    deltaNodes += 1
    parent.wrap(newTag)
    for i in range(orig_sibling_num - delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        parent.insert_after(newTag)


def SetFirstParentSiblingTagName(tagName):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    newTag = soup.new_tag(tagName)
    newTag.attrs["hidden"] = ""
    deltaNodes += 1
    parent.insert_before(newTag)


def RemoveFirstParentSiblingAdAttribute():
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    newTag = soup.new_tag('p')
    newTag.attrs["hidden"] = ""
    deltaNodes += 1
    parent.insert_before(newTag)


def IncreaseFirstParentInboundConnections(delta):
    global deltaNodes, soup
    tag = soup.find("body")
    for i in range(delta):
        script_tag = soup.new_tag('script')
        script_tag.append('var x = document.querySelector(\'[src="' + requestURL +
                          '"]\'); x.classList.add(\'eirqghjuijsiodvpasfwegg\'); x.classList.remove(\'eirqghjuijsiodvpasfwegg\');')
        deltaNodes += 1
        tag.append(script_tag)


def IncreaseFirstParentOutboundConnections(delta):
    # add children to first parent
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        parent.append(newTag)


def IncreaseFirstParentInboundOutboundConnections(delta):
    # add children to first parent
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        parent.append(newTag)


def IncreaseFirstParentAverageDegreeConnectivity(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    newTag = soup.new_tag('div')
    newTag.attrs["hidden"] = ""
    deltaNodes += 1
    parent.append(newTag)
    for i in range(delta):
        newTag2 = soup.new_tag('p')
        newTag2.attrs["hidden"] = ""
        deltaNodes += 1
        newTag.append(newTag2)


def DecreaseFirstParentAverageDegreeConnectivity(delta):
    global deltaNodes, soup
    tag = soup.find("body")
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        tag.append(newTag)


def featureMapbacks(name, html, url, delta=None):
    global deltaNodes, soup, requestURL
    deltaNodes = 0
    soup = html
    # input(type(soup))
    requestURL = url

    before_mapback = str(soup)

    if name == "FEATURE_GRAPH_NODES" or name == "FEATURE_GRAPH_EDGES":
        addNodes(delta)
    elif name == "FEATURE_INBOUND_OUTBOUND_CONNECTIONS":
        addNodes(delta, False)
    elif name == "FEATURE_ASCENDANTS_AD_KEYWORD":  # can only turn this feature from true to false
        wrapNodes(3)
    elif "FEATURE_FIRST_PARENT_TAG_NAME" in name:
        tagName = re.split("=", name)[1]
        wrapNodes(1, tagName)
    elif name == "FEATURE_FIRST_NUMBER_OF_SIBLINGS":
        if delta > 0:
            IncreaseFirstNumberOfSiblings(delta)
        elif delta < 0:
            DecreaseFirstNumberOfSiblings(-delta)
    elif name == "FEATURE_FIRST_PARENT_NUMBER_OF_SIBLINGS":
        if delta > 0:
            IncreaseFirstParentNumberOfSiblings(delta)
        elif delta < 0:
            DecreaseFirstParentNumberOfSiblings(-delta)
    elif "FEATURE_FIRST_PARENT_SIBLING_TAG_NAME" in name:
        tagName = re.split("=", name)[1]
        SetFirstParentSiblingTagName(tagName)
    elif name == "FEATURE_FIRST_PARENT_SIBLING_AD_ATTRIBUTE":
        if delta > 0:
            print("Invalid delta parameter.")
        elif delta < 0:
            RemoveFirstParentSiblingAdAttribute()
    elif name == "FEATURE_FIRST_PARENT_INBOUND_CONNECTIONS":
        if delta > 0:
            IncreaseFirstParentInboundConnections(delta)
        else:
            print("Invalid delta parameter.")
    elif name == "FEATURE_FIRST_PARENT_OUTBOUND_CONNECTIONS":
        if delta > 0:
            IncreaseFirstParentOutboundConnections(delta)
        else:
            print("Invalid delta parameter.")
    elif name == "FEATURE_FIRST_PARENT_INBOUND_OUTBOUND_CONNECTIONS":
        if delta > 0:
            IncreaseFirstParentInboundOutboundConnections(delta)
        else:
            print("Invalid delta parameter.")
    elif name == "FEATURE_FIRST_PARENT_AVERAGE_DEGREE_CONNECTIVITY":
        if delta > 0:
            IncreaseFirstParentAverageDegreeConnectivity(delta)
        elif delta < 0:
            DecreaseFirstParentAverageDegreeConnectivity(-1 * delta)

    after_mapback = str(soup)
    if before_mapback == after_mapback:
        return None
    else:
        print("Modification occured!")

    return soup
