def try_load_jsonschema():
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        raise ImportError(
            "To deserialize objects, install polliwog with the serialization extra: "
            + "`pip install entente[serialization]`"
        )
    return jsonschema


def validator_for(schema_path, ref):
    import simplejson as json

    jsonschema = try_load_jsonschema()

    with open(schema_path, "r") as f:
        schema = json.load(f)

    resolver = jsonschema.RefResolver.from_schema(schema)
    return jsonschema.Draft7Validator({"$ref": ref}, resolver=resolver)
