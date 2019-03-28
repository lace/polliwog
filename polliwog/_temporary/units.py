class Units(object):
    def __init__(self):
        import numpy as np

        self._units = units = {}

        # lengths: conversion to m
        units["m"] = units["meter"] = units["meters"] = ("length", 1.0)
        units["mm"] = units["millimeter"] = units["millimeters"] = (
            "length",
            units["m"][1] / 1000,
        )
        units["cm"] = units["centimeter"] = units["centimeters"] = (
            "length",
            units["m"][1] / 100,
        )
        units["in"] = units["inch"] = units["inches"] = ("length", 0.0254)
        units["ft"] = units["foot"] = units["feet"] = ("length", 0.3048)
        units["fathoms"] = ("length", 1.8288)
        units["cubits"] = ("length", 0.4572)

        # weights: conversion to kg
        units["kg"] = units["kilograms"] = ("weight", 1.0)
        units["g"] = units["grams"] = ("weight", units["kg"][1] / 1000)
        units["lbs"] = units["pounds"] = units["lb"] = (
            "weight",
            units["kg"][1] / 2.20462,
        )
        units["stone"] = ("weight", units["kg"][1] / 0.157473)

        # angles: conversion to degrees
        units["deg"] = units["degrees"] = ("angle", 1.0)
        units["rad"] = units["radians"] = ("angle", units["deg"][1] * 180 / np.pi)

        # time: conversion to seconds
        units["sec"] = units["second"] = units["seconds"] = ("time", 1.0)
        units["min"] = units["minute"] = units["minutes"] = (
            "time",
            units["sec"][1] * 60,
        )
        units["hr"] = units["hour"] = units["hours"] = ("time", 60 * units["min"][1])
        units["day"] = units["days"] = ("time", units["hour"][1] * 24)
        units["yr"] = units["year"] = units["years"] = (
            "time",
            units["day"][1] * 365.242,
        )

        units["hours_per_week"] = ("time_rate", 1.0)

        self._default_units = {
            "metric": {
                "length": "cm",
                "weight": "kg",
                "angle": "deg",
                "time": "yr",
                "time_rate": "hours_per_week",
            },
            "united_states": {
                "length": "in",
                "weight": "lb",
                "angle": "deg",
                "time": "yr",
                "time_rate": "hours_per_week",
            },
        }

    @property
    def all_units(self):
        """
        Return a list of all supported units.

        """
        return self._units.keys()

    @property
    def all_units_classes(self):
        """
        Return a list of all supported units classes, e.g.
        'length', 'weight', 'angle, 'time', 'time_rate'.

        """
        return list(set([item[0] for item in self._units.values()]))

    def units_class(self, units):
        """
        Returns 'length', 'weight', 'angle', 'time', or 'time_rate'.

            >>> units_class('cm')
            'length'

        """
        if units is None or units == "":
            return None
        return self._units[units][0]

    def units_in_class(self, uclass):
        """
        Return a list of all units in uclass, where uclass is e.g.
        length', 'weight', 'angle, 'time', 'time_rate'.
        """
        return [key for key, (uclass_0, _) in self._units.items() if uclass_0 == uclass]

    @property
    def lengths(self):
        """
        List of all supported lengths.
        """
        return self.units_in_class("length")

    @property
    def weights(self):
        """
        List of all supported weights.
        """
        return self.units_in_class("weight")

    @property
    def angles(self):
        """
        List of all supported angles.
        """
        return self.units_in_class("angle")

    @property
    def times(self):
        """
        List of all supported times.
        """
        return self.units_in_class("time")

    @property
    def time_rates(self):
        """
        List of all supported time rates.
        """
        return self.units_in_class("time_rate")

    def default_units(self, unit_system="metric", exceptions={}):
        """
        Get the default unit for a given unit system. unit_class is
        'length', 'weight', 'angle', or 'time'. unit_system is
        either 'metric' or 'united_states'.

        """
        return dict(
            list(self._default_units[unit_system].items()) + list(exceptions.items())
        )

    def raw(self, units):
        """
        Returns a raw units conversion factor. Try not to use this.
        Use factor() or convert() instead.

        """
        return self._units[units][1]

    def factor(
        self, from_units, to_units, units_class=None
    ):  # FIXME pylint: disable=redefined-outer-name
        """
        Return a conversion factor:

            >>> value_in_cm = 25
            >>> value_in_cm * factor('cm', 'mm')
            250

        class: If specified, the class of the units must match the class provided.

        """
        if (from_units is None or not len(from_units)) and (
            to_units is None or not len(to_units)
        ):  # pylint: disable=len-as-condition
            return 1.0
        if from_units == to_units:
            return 1.0
        if self._units[from_units][0] != self._units[to_units][0]:
            raise ValueError(
                "Can't convert between apples and oranges (%s and %s)"
                % (from_units, to_units)
            )
        if units_class and self._units[from_units][0] != units_class:
            raise ValueError(
                "Units class must be %s, but got %s"
                % (units_class, self._units[from_units][0])
            )
        return self._units[from_units][1] / self._units[to_units][1]

    def convert(
        self, value, from_units, to_units, units_class=None
    ):  # FIXME pylint: disable=redefined-outer-name
        """
        Convert a number from one unit to another.

        class: If specified, the class of the units must match the class provided.

        Returns a tuple with the converted value and the units.

            >>> value_cm = 25
            >>> value, units = convert(value_cm, 'cm', 'mm')
            >>> value
            250
            >>> units
            'mm'

        """
        # Get factor first so we return errors for apples and oranges,
        # even when value is None
        factor = self.factor(
            from_units, to_units, units_class=units_class
        )  # FIXME pylint: disable=redefined-outer-name
        if value is None:
            return None, None
        return value * factor, to_units

    def convert_list(self, a_list, from_units, to_units):
        """
        Convenience helper to convert a list of numbers from one unit to another.

        Unlike convert(), does not return a tuple.

            >>> convert_list([10, 20, 30], 'cm', 'mm')
            [100, 200, 300]

        """
        factor = self.factor(
            from_units, to_units
        )  # FIXME pylint: disable=redefined-outer-name
        return [factor * x for x in a_list]

    def convert_to_default(self, value, from_units, defaults):
        """
        Convert a number from the given unit to a default for
        the given unit system.

        Returns a tuple with the converted value and the units.

            >>> value_cm = 100
            >>> convert_to_default(value_cm, 'cm', {'length': 'in', 'weight': 'lb', 'angle': 'deg', 'time': 'yr'})
            (39.3701, 'in')

        """
        units_class = self.units_class(
            from_units
        )  # FIXME pylint: disable=redefined-outer-name
        if units_class:
            to_units = defaults[units_class]
            return self.convert(value, from_units, to_units)
        else:
            return value, from_units

    def convert_to_system_default(self, value, from_units, to_unit_system="metric"):
        """
        Convert a number from the given unit to a default for
        the given unit system.

        Returns a tuple with the converted value and the units.

            >>> value_cm = 100
            >>> convert_to_system_default(value_cm, 'cm', 'united_states')
            (39.3701, 'in')

        """
        return self.convert_to_default(
            value, from_units, self.default_units(to_unit_system)
        )

    def prettify(self, value, units, precision=None):
        """
        Take a value, and units, and return rounded values in the
        default metric and United States units.

        The default precision values vary per unit. Specifying an
        integer for `precision` will override those defaults.

            >>> prettify(182.13992, 'cm'),
            (182.0, 'cm', 71.75, 'in')
            >>> prettify(182.13992, 'cm', precision=1)
            (182.1, 'cm', 71.7, 'in')
            >>> prettify(182.13992, 'cm', precision=2)
            (182.14, 'cm', 71.71, 'in')

        """

        def round_value(value, units):
            from .rounding import round_to

            if precision is None:
                nearest = {"cm": 0.5, "in": 0.25, "kg": 0.5, "lb": 1.0, "yr": 1.0}.get(
                    units, 0.1
                )
            else:
                # Work around weird floating point rounding issues
                nearest = 1.0 / 10 ** precision
            return round_to(value, nearest), units

        return round_value(*self.convert_to_system_default(value, units)) + round_value(
            *self.convert_to_system_default(value, units, "united_states")
        )


_units = Units()

all_units = _units.all_units
all_units_classes = _units.all_units_classes
units_class = _units.units_class
units_in_class = _units.units_in_class
lengths = _units.lengths
weights = _units.weights
angles = _units.angles
times = _units.times
time_rates = _units.time_rates
default_units = _units.default_units
convert_to_default = _units.convert_to_default
raw = _units.raw
factor = _units.factor
convert = _units.convert
convert_list = _units.convert_list
convert_to_system_default = _units.convert_to_system_default
prettify = _units.prettify
