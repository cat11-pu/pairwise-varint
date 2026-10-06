"""Behaviour tests for the variable length integer kernel.

The tests only describe what a caller of the kernel is allowed to observe:
the bytes the encoders put out, the numbers and sizes the decoders return, and
the errors the buffer helpers report.  Run them from the project root:

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import varint.core as varint


class RoundTripTests(unittest.TestCase):
    """Values that have to survive a trip through the wire form."""

    def test_small_values_round_trip_and_use_one_byte(self):
        for value in (0, 1, 42, 126, 127):
            wire = varint.encode_uvarint(value)
            self.assertEqual(len(wire), 1)
            self.assertEqual(varint.decode_uvarint(wire), (value, 1))
        self.assertEqual(varint.encode_uvarint(0), b"\x00")
        self.assertEqual(varint.encode_uvarint(127), b"\x7f")

    def test_multi_group_values_round_trip_on_the_wire(self):
        for value in (128, 300, 16384, 1 << 20, (1 << 32) + 5):
            wire = varint.encode_uvarint(value)
            self.assertEqual(varint.decode_uvarint(wire), (value, len(wire)))
        self.assertEqual(varint.encode_uvarint(128), b"\x80\x01")
        self.assertEqual(varint.encode_uvarint(300), b"\xac\x02")
        self.assertEqual(varint.encode_uvarint(16384), b"\x80\x80\x01")

    def test_the_64_bit_boundary_round_trips_and_wider_forms_are_refused(self):
        wire = varint.encode_uvarint(varint.UINT64_MAX)
        self.assertEqual(len(wire), varint.MAX_VARINT_BYTES)
        self.assertEqual(
            varint.decode_uvarint(wire),
            (varint.UINT64_MAX, varint.MAX_VARINT_BYTES),
        )
        self.assertRaises(
            varint.VarintError, varint.decode_uvarint, b"\xff" * 9 + b"\x7f"
        )
        self.assertRaises(
            varint.VarintError, varint.decode_uvarint, b"\x80" * 9 + b"\x02"
        )
        self.assertRaises(
            varint.VarintError, varint.decode_uvarint, b"\x80" * 10 + b"\x01"
        )

    def test_a_value_has_exactly_one_encoding(self):
        wire = varint.encode_uvarint(128)
        self.assertEqual(varint.decode_uvarint(wire), (128, len(wire)))
        self.assertRaises(varint.VarintError, varint.decode_uvarint, b"\x80\x00")
        self.assertRaises(varint.VarintError, varint.decode_uvarint, b"\x81\x00")
        self.assertRaises(
            varint.VarintError, varint.decode_uvarint, b"\x80\x80\x00"
        )

    def test_signed_values_round_trip_including_negative_ones(self):
        for value in (-65, -1, -2, 0, 1, 63, 64, varint.INT64_MIN, varint.INT64_MAX):
            wire = varint.encode_svarint(value)
            self.assertEqual(varint.decode_svarint(wire), (value, len(wire)))
        self.assertEqual(varint.encode_svarint(-1), b"\x01")
        self.assertEqual(varint.encode_svarint(1), b"\x02")

    def test_the_big_endian_group_order_round_trips(self):
        for value in (0, 127, 128, 300, 16384, varint.UINT64_MAX):
            wire = varint.encode_uvarint_be(value)
            self.assertEqual(varint.decode_uvarint_be(wire), (value, len(wire)))
        self.assertEqual(varint.encode_uvarint_be(128), b"\x81\x00")
        self.assertEqual(varint.encode_uvarint_be(300), b"\x82\x2c")
        self.assertEqual(varint.decode_uvarint_be(b"\x82\x2c"), (300, 2))
        self.assertRaises(
            varint.VarintError, varint.decode_uvarint_be, b"\x80\x00"
        )
        self.assertRaises(
            varint.VarintError,
            varint.decode_uvarint_be,
            b"\x82" + b"\xff" * 8 + b"\x7f",
        )


class SizeTests(unittest.TestCase):
    """The reported size has to be the size that comes out of the encoder."""

    def test_the_encoded_size_matches_the_bytes_that_come_out(self):
        for value in (0, 1, 127, 128, 300, 16383, 16384, 1 << 20, varint.UINT64_MAX):
            self.assertEqual(
                varint.encoded_size(value), len(varint.encode_uvarint(value))
            )
        self.assertEqual(varint.encoded_size(0), 1)
        self.assertEqual(varint.encoded_size(127), 1)
        self.assertEqual(varint.encoded_size(128), 2)
        self.assertEqual(varint.encoded_size(16383), 2)
        self.assertEqual(varint.encoded_size(16384), 3)


class BufferTests(unittest.TestCase):
    """Buffers that carry several values in a row."""

    def test_a_whole_buffer_of_values_decodes_in_one_pass(self):
        self.assertEqual(varint.decode_many(b""), [])
        buffer = b"".join(
            varint.encode_uvarint(value) for value in (3, 300, 7, 127)
        )
        self.assertEqual(varint.decode_many(buffer), [3, 300, 7, 127])
        self.assertRaises(varint.VarintError, varint.decode_many, b"\x03\x85")
        self.assertEqual(varint.decode_many(b"\x03\x05"), [3, 5])

    def test_the_reader_keeps_partial_values_until_the_next_feed(self):
        reader = varint.VarintReader()
        reader.feed(b"\x03\x85")
        self.assertEqual(reader.read_value(), 3)
        self.assertEqual(reader.pending(), 1)
        self.assertIsNone(reader.read_value())
        self.assertEqual(reader.pending(), 1)
        reader.feed(b"\x02")
        self.assertEqual(reader.read_value(), 261)
        self.assertEqual(reader.pending(), 0)
        self.assertIsNone(reader.read_value())
        reader.feed(b"\x80" * 10)
        self.assertRaises(varint.VarintError, reader.read_value)
        self.assertEqual(reader.pending(), 10)
        reader.reset()
        self.assertEqual(reader.pending(), 0)


class RejectionTests(unittest.TestCase):
    """Inputs that must be refused instead of being half accepted."""

    def test_bad_inputs_are_refused_instead_of_half_accepted(self):
        self.assertRaises(varint.VarintError, varint.encode_uvarint, -1)
        self.assertRaises(
            varint.VarintError, varint.encode_uvarint, varint.UINT64_MAX + 1
        )
        self.assertRaises(varint.VarintError, varint.encode_uvarint_be, -1)
        self.assertRaises(varint.VarintError, varint.encoded_size, -1)
        self.assertRaises(varint.VarintError, varint.decode_uvarint, b"")
        with self.assertRaises(varint.VarintError) as caught:
            varint.decode_uvarint(b"\x80\x84\x80")
        self.assertTrue(caught.exception.incomplete)
        with self.assertRaises(varint.VarintError) as too_long:
            varint.decode_uvarint(b"\x80" * 10)
        self.assertFalse(too_long.exception.incomplete)


if __name__ == "__main__":
    unittest.main()
