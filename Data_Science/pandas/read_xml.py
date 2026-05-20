from dataframe import DataFrame
import pandas as pd
import xml.etree.ElementTree as ET

def get_nested_tags_text(tag):
    if len(tag):
        res = {}
        for t in tag:
            res.update(get_nested_tags_text(t))
        return res
    return {tag.tag: tag.text}

def add_to_data(data, key, val):
    if key not in data:
        data[key] = []
    data[key].append(val)

def read_xml(fname, record_tag=None):
    tree = ET.parse(fname)
    root = tree.getroot()

    if record_tag is None:
        if len(root):
            record_tag = root[0].tag
        else:
            record_tag = root.tag

    data = {}
    for record in root.findall(record_tag):
        for col, val in record.attrib.items():
            norm_val = pd.to_numeric(val, errors='ignore')
            add_to_data(data, col, norm_val)

        for item in record:
            tag_name = item.tag 
            val = item.text

            if val is None or val.strip():
                norm_val = pd.to_numeric(val, errors='ignore')
                add_to_data(data, tag_name, norm_val)
            else:
                add_to_data(data, tag_name, get_nested_tags_text(item))
    
    return DataFrame(data)

# print(read_xml('examples/xml1.xml'))

# print(read_xml('examples/xml2.xml'))

# print(read_xml('examples/xml3.xml'))

print(read_xml('examples/xml4.xml'))

