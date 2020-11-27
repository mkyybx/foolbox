#!/usr/bin/env python3

import random
import re
import string
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from fuzzywuzzy import fuzz

# process features
deltaNodes = 0  # # of edges should be equal to # of nodes
soup = None
requestURL = None

adKeyWord = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser",
             "advertise", "redirect", "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid",
             "pb.min", "affiliate", "ban", "delivery", "promo", "tag", "zoneid", "siteid", "pageid", "size", "viewid",
             "zone_id", "google_afc", "google_afs"]
adKeyChar = [".", "/", "&", "=", ";", "-", "_", "/", "*", "^", "?", ";", "|", ","]
screenResolution = ["screenheight", "screenwidth", "browserheight", "browserwidth", "screendensity", "screen_res",
                    "screen_param", "screenresolution", "browsertimeoffset"]


def BSFilterByAnyString(tag):
    for attr_val in list(tag.attrs.values()):
        if fuzz.ratio(attr_val, requestURL) > 90:
            return tag
    # if requestURL in tag.attrs.values():
    #    return tag


def getAttrByURL(tag):
    for attr in list(tag.attrs.keys()):
        if fuzz.ratio(tag.attrs[attr], requestURL) > 90:
            return attr


def IncreaseURLLength(delta):
    if delta < 0:
        print("changeNodes with delta < 0!")
        return
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    for i in range(delta):
        tag.attrs[attr] += "#"


# All of the following "Replacing" would replace with random chars. The server may parse the URL dynamically
def ModifyADKeyword(add):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        for adWord in adKeyWord:
            randStr = ""
            for _ in adWord:
                randStr += random.choice(string.ascii_lowercase)
            tag.attrs[attr] = tag.attrs[attr].replace(adWord, randStr)
    elif add == 1:
        tag.attrs[attr] += "#" + random.choice(adKeyWord)


# Possibly not usable since it replaces the special chars in the URL, but the server may parse the URL dynamically
def ModifyADKeyChar(add):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        for adWord in adKeyWord:
            for adChar in adKeyChar:
                combinedStr = adChar + adWord
                tag.attrs[attr] = tag.attrs[attr].replace(combinedStr, random.choice(string.ascii_lowercase) + adWord)
    elif add == 1:
        tag.attrs[attr] += "#?" + random.choice(adKeyWord)


# Not tested
def ModifySemicolon(add):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        tag.attrs[attr] = tag.attrs[attr].replace(";", "&")
    elif add == 1:
        tag.attrs[attr] += "#;"


def ModifyQueryString(removeQS):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if removeQS == 1:
        tag.attrs[attr] = tag.attrs[attr].replace("?", random.choice(string.ascii_lowercase))
    elif removeQS == -1:
        tag.attrs[attr] += "?" + random.choice(string.ascii_lowercase) + "=" + random.choice(
            string.ascii_lowercase)


def GetQSDictFromURL(url):
    urlParts = urlparse(url)
    qsdict = parse_qs(urlParts.query)
    for key in qsdict.keys():
        qsdict[key] = qsdict[key][0]
    return qsdict


def GenerateURLWithQS(url, qsdict):
    urlParts = urlparse(url)
    urlPartsList = list(urlParts)
    urlPartsList[4] = urlencode(qsdict)
    return urlunparse(urlPartsList)


def ModifyBaseDomainQueryString(add, domain):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    # Trim the domain
    if domain.startswith("www."):
        domain = domain[4:]
    if add == -1:
        qsdict = GetQSDictFromURL(tag.attrs[attr])
        # Find the key/val pair
        for key in qsdict.keys():
            if domain == qsdict[key]:
                # Remove only if we found, prevent crash
                del qsdict[key]
                break
        # Write back
        tag.attrs[attr] = GenerateURLWithQS(tag.attrs[attr], qsdict)
    elif add == 1:
        qsdict = GetQSDictFromURL(tag.attrs[attr])
        qsdict[str(random.randint(0, 999999))] = domain
        tag.attrs[attr] = GenerateURLWithQS(tag.attrs[attr], qsdict)


def ModifyScreenDimensionInBaseURL(add):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        tag.attrs[attr] = ReplaceX(tag.attrs[attr])
    elif add == 1:
        tag.attrs[attr] += "#" + str(random.randint(0, 10000)) + "x" + str(random.randint(0, 10000))


def ModifyADDimensionInQS(add):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        qsdict = GetQSDictFromURL(tag.attrs[attr])
        for key in qsdict.keys():
            qsdict[key] = ReplaceX(qsdict[key])
        tag.attrs[attr] = GenerateURLWithQS(tag.attrs[attr], qsdict)

    elif add == 1:
        if "?" in tag.attrs[attr]:
            tag.attrs[attr] += "&" + random.choice(string.ascii_lowercase) + "=" + str(
                random.randint(0, 10000)) + "x" + str(random.randint(0, 10000))
        else:
            tag.attrs[attr] += "?" + random.choice(string.ascii_lowercase) + "=" + str(
                random.randint(0, 10000)) + "x" + str(random.randint(0, 10000))


def ModifyScreenDimensionKeywordInQS(add):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        qsdict = GetQSDictFromURL(tag.attrs[attr])
        for key in qsdict.keys():
            if key in screenResolution:
                del qsdict[key]
                break
        tag.attrs[attr] = GenerateURLWithQS(tag.attrs[attr], qsdict)

    elif add == 1:
        if "?" in tag.attrs[attr]:
            tag.attrs[attr] += "&" + random.choice(screenResolution) + "=" + random.choice(string.ascii_lowercase)
        else:
            tag.attrs[attr] += "?" + random.choice(screenResolution) + "=" + random.choice(string.ascii_lowercase)
    print(tag)


def ReplaceX(text):
    splitedStr = re.split("\\d{2,4}[xX_-]\\d{2,4}", text)
    matchedStr = re.findall("\\d{2,4}[xX_-]\\d{2,4}", text)
    text = ""
    for i in range(len(matchedStr)):
        replacingStr = re.sub("[xX_-]", random.choice(string.ascii_lowercase), matchedStr[i])
        text += splitedStr[i] + replacingStr
    # if len(matchedStr) > 0:
    text += splitedStr[len(splitedStr) - 1]
    return text


def ModifyHostName(setToBaseDomain, baseDomain):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if setToBaseDomain == 1:
        urlParts = urlparse(tag.attrs[attr])
        urlPartsList = list(urlParts)
        urlPartsList[1] = baseDomain
        tag.attrs[attr] = urlunparse(urlPartsList)
    elif setToBaseDomain == -1:
        urlParts = urlparse(tag.attrs[attr])
        urlPartsList = list(urlParts)
        urlPartsList[1] = urlPartsList[1] + "."
        tag.attrs[attr] = urlunparse(urlPartsList)


def ModifyHostNameWithSubDomain(addSubdomain, baseDomain):
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if addSubdomain == 1:
        urlParts = urlparse(tag.attrs[attr])
        urlPartsList = list(urlParts)
        urlPartsList[1] = baseDomain + "." + urlPartsList[1]
        tag.attrs[attr] = urlunparse(urlPartsList)
    elif addSubdomain == -1:
        urlParts = urlparse(tag.attrs[attr])
        urlPartsList = list(urlParts)
        # Trim the domain
        if baseDomain.startswith("www."):
            baseDomain = baseDomain[4:]
        urlPartsList[1] = urlPartsList[1].replace(baseDomain + ".", "").replace("." + baseDomain, "").replace(
            baseDomain, "")
        tag.attrs[attr] = urlunparse(urlPartsList)

    print(tag)


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


def IncreaseURLLengt(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    tag.attrs["src"] = tag.attrs["src"] + '*' * delta


def featureMapbacks(name, html, url, delta=None, domain=None):
    global deltaNodes, soup, requestURL
    deltaNodes = 0
    soup = html
    requestURL = url

    before_mapback = str(soup)
    print("Feature name: %s | %d" % (name, delta))

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
        elif delta < 0:
            print("Invalid delta parameter.")
    elif name == "FEATURE_FIRST_PARENT_OUTBOUND_CONNECTIONS":
        if delta > 0:
            IncreaseFirstParentOutboundConnections(delta)
        elif delta < 0:
            print("Invalid delta parameter.")
    elif name == "FEATURE_FIRST_PARENT_INBOUND_OUTBOUND_CONNECTIONS":
        if delta > 0:
            IncreaseFirstParentInboundOutboundConnections(delta)
        elif delta < 0:
            print("Invalid delta parameter.")
    elif name == "FEATURE_FIRST_PARENT_AVERAGE_DEGREE_CONNECTIVITY":
        if delta > 0:
            IncreaseFirstParentAverageDegreeConnectivity(delta)
        elif delta < 0:
            DecreaseFirstParentAverageDegreeConnectivity(-1 * delta)

    # URL Features
    # TODO: lack ctx info for QS and URL length. We can add it if we know we are modifying the features of the same webpage. We know for sure some features would change if modifying other features

    elif name == "FEATURE_URL_LENGTH":
        if delta > 0:
            IncreaseURLLength(delta)
        elif delta < 0:
            print("Invalid delta parameter.")
    elif name == "FEATURE_AD_KEYWORD":
        ModifyADKeyword(delta)
    elif name == "FEATURE_SPECIAL_CHAR_AD_KEYWORD":
        ModifyADKeyChar(delta)
    elif name == "FEATURE_SEMICOLON_PRESENT":
        ModifySemicolon(delta)
    elif name == "FEATURE_VALID_QS":
        ModifyQueryString(delta)
    elif name == "FEATURE_BASE_DOMAIN_IN_QS":
        ModifyBaseDomainQueryString(delta, domain)
    elif name == "FEATURE_AD_DIMENSIONS_IN_QS":
        ModifyADDimensionInQS(delta)
    elif name == "FEATURE_SCREEN_DIMENSIONS_IN_QS":
        ModifyScreenDimensionKeywordInQS(delta)
    elif name == "FEATURE_AD_DIMENSIONS_IN_COMPLETE_URL":
        ModifyScreenDimensionInBaseURL(delta)
    elif name == "FEATURE_DOMAIN_PARTY":
        ModifyHostName(delta, domain)
    elif name == "FEATURE_SUB_DOMAIN_CHECK":
        ModifyHostNameWithSubDomain(delta, domain)

    after_mapback = str(soup)
    if before_mapback == after_mapback:
        return None
    else:
        print("Modification occured!")

    return soup
