# Reference: https://github.com/wataru0/gym_custom_terrain
import os
import shutil
import xml.etree.ElementTree as Et

def get_image_abs_path(image_path: str) -> str:
    if image_path.startswith("/"):
        return image_path
    cwd = os.getcwd()
    return os.path.join(cwd, image_path)


def generate_terrain(xml_name: str, terrain_image_name: str):
    cwd = os.getcwd()
    xml_path = os.path.join(cwd, xml_name)

    tmp_xml_file = Et.parse(xml_path)
    root = tmp_xml_file.getroot()

    image_abs_path = get_image_abs_path(terrain_image_name)

    for child in root:
        if child.tag == "asset":
            hfield = child.find("hfield")
            hfield.set("file", image_abs_path)

    tmp_xml_file.write(xml_path, encoding="UTF-8")
