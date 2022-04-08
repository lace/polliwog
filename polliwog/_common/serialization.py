def try_load_jsonschema_and_simplejson():
    try:
        import jsonschema
        import simplejson
    except ImportError:  # pragma: no cover
        raise ImportError(
            "To deserialize objects, install polliwog with the serialization extra: "
            + "`pip install entente[serialization]`"
        )
    return jsonschema, simplejson


def validator_for(schema_path, ref):
    jsonschema, json = try_load_jsonschema_and_simplejson()

    with open(schema_path, "r") as f:
        schema = json.load(f)

    resolver = jsonschema.RefResolver.from_schema(schema)
    return jsonschema.Draft7Validator({"$ref": ref}, resolver=resolver)
