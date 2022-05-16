import simplejson as json

from .pathlib import project_relative_path, SCHEMA_PATH


def test_schema_matches_types_package():
    with open(SCHEMA_PATH, "r") as f:
        schema_from_project = json.load(f)

    with open(
        project_relative_path("types", "src", "generated", "schema.json"), "r"
    ) as f:
        schema_from_types_package = json.load(f)

    assert schema_from_project == schema_from_types_package
