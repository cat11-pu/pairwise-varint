"""Variable length integer kernel built on the standard library only.

A varint carries an integer in seven bit groups: each byte holds seven payload
bits and a continuation bit, and the byte that ends a value is the one whose
continuation bit is clear.  Two group orders are supported:

* the little endian order (:func:`encode_uvarint`, :func:`decode_uvarint`)
  writes the least significant group first, so every following group is
  shifted seven bits further left before it is merged in;
* the big endian order (:func:`encode_uvarint_be`, :func:`decode_uvarint_be`)
  writes the most significant group first, so the groups already seen are
  shifted seven bits left before the next one is merged in.

Unsigned values stop at 64 bits: ten bytes is the longest varint, and the
group that carries the top of a 64 bit value may use only one of its seven
payload bits.  Signed values travel through the zigzag map before they are
encoded, so -1 lands on the one byte value 1.

Everything stays in memory: callers hand bytes to the decode helpers or feed
them to a :class:`VarintReader` and get integers back.
"""

#: Longest varint a 64 bit value may need.
MAX_VARINT_BYTES = 10

#: Largest unsigned value the helpers carry.
UINT64_MAX = 2 ** 64 - 1

#: Signed range the zigzag helpers accept.
INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1

#: Continuation bit of a varint byte.
CONTINUATION = 0x80

#: Seven payload bits carried by a varint byte.
PAYLOAD_MASK = 0x7F

#: Payload bits the byte holding the top of a 64 bit value may use.
FINAL_BYTE_MAX = 0x01


class VarintError(Exception):
    """Raised when bytes cannot be read as a varint.

    ``incomplete`` is set when the bytes seen so far could still grow into a
    varint: a streaming caller should wait for more bytes instead of
    reporting a broken stream.
    """

    def __init__(self, message, incomplete=False):
        super().__init__(message)
        self.message = message
        self.incomplete = incomplete


def _check_unsigned(value, where="value"):
    """Refuse a value outside the unsigned 64 bit range."""
    if not 0 <= value <= UINT64_MAX:
        raise VarintError(
            "%s %r is outside the unsigned 64 bit range" % (where, value)
        )


# ---------------------------------------------------------------------------
# Little endian seven bit groups
# ---------------------------------------------------------------------------

def encode_uvarint(value):
    """Encode value with its least significant seven bit group first."""
    _check_unsigned(value)
    chunk = bytearray()
    while True:
        byte = value & PAYLOAD_MASK
        value >>= 7
        if value:
            chunk.append(byte | CONTINUATION)
            continue
        chunk.append(byte)
        return bytes(chunk)


def decode_uvarint(data, offset=0):
    """Decode the little endian varint at offset, returning (value, size)."""
    if offset < 0:
        raise VarintError("offset %d is negative" % offset)
    value = 0
    shift = 0
    pos = offset
    while True:
        if pos >= len(data):
            raise VarintError("the varint needs more bytes", incomplete=True)
        byte = data[pos]
        pos += 1
        if pos - offset == MAX_VARINT_BYTES:
            if byte & CONTINUATION:
                raise VarintError(
                    "the varint is longer than %d bytes" % MAX_VARINT_BYTES
                )
            if byte > FINAL_BYTE_MAX:
                raise VarintError("the varint does not fit 64 bits")
        value |= (byte & PAYLOAD_MASK) << shift
        if not byte & CONTINUATION:
            if pos - offset > 1 and not byte & PAYLOAD_MASK:
                raise VarintError("the varint is not the shortest form")
            return value, pos - offset
        shift += 7


def encode_svarint(value):
    """Encode a signed 64 bit integer as a zigzag varint."""
    if not INT64_MIN <= value <= INT64_MAX:
        raise VarintError(
            "the signed value %r is outside the 64 bit range" % (value,)
        )
    return encode_uvarint((value << 1) ^ (value >> 63))


def decode_svarint(data, offset=0):
    """Decode the zigzag varint at offset, returning (value, size)."""
    raw, size = decode_uvarint(data, offset)
    return (raw >> 1) ^ -(raw & 1), size


# ---------------------------------------------------------------------------
# Big endian seven bit groups
# ---------------------------------------------------------------------------

def encode_uvarint_be(value):
    """Encode value with its most significant seven bit group first."""
    _check_unsigned(value)
    groups = []
    while True:
        groups.append(value & PAYLOAD_MASK)
        value >>= 7
        if not value:
            break
    groups.reverse()
    chunk = bytearray()
    for group in groups[:-1]:
        chunk.append(group | CONTINUATION)
    chunk.append(groups[-1])
    return bytes(chunk)


def decode_uvarint_be(data, offset=0):
    """Decode the big endian varint at offset, returning (value, size)."""
    if offset < 0:
        raise VarintError("offset %d is negative" % offset)
    groups = []
    pos = offset
    while True:
        if pos >= len(data):
            raise VarintError("the varint needs more bytes", incomplete=True)
        byte = data[pos]
        pos += 1
        groups.append(byte & PAYLOAD_MASK)
        if len(groups) == MAX_VARINT_BYTES and byte & CONTINUATION:
            raise VarintError(
                "the varint is longer than %d bytes" % MAX_VARINT_BYTES
            )
        if not byte & CONTINUATION:
            break
    if len(groups) == MAX_VARINT_BYTES and groups[0] > FINAL_BYTE_MAX:
        raise VarintError("the varint does not fit 64 bits")
    if len(groups) > 1 and not groups[0]:
        raise VarintError("the varint is not the shortest form")
    value = 0
    for group in groups:
        value = (value << 7) | group
    return value, pos - offset


# ---------------------------------------------------------------------------
# Sizes, buffers and streaming
# ---------------------------------------------------------------------------

def encoded_size(value):
    """Number of bytes the little endian encoding of value occupies."""
    _check_unsigned(value)
    return (value.bit_length() + 6) // 7 or 1


def decode_many(data):
    """Decode every varint held by data, in order.

    The whole buffer has to be a run of complete values: a buffer that ends in
    the middle of a value is reported instead of being cut short.
    """
    values = []
    offset = 0
    while offset < len(data):
        value, size = decode_uvarint(data, offset)
        values.append(value)
        offset += size
    return values


class VarintReader:
    """Incremental reader for a stream of varints.

    ``feed`` appends whatever arrived and ``read_value`` returns the next
    complete value, or None while the buffer holds only part of one.  Bytes
    that cannot be a varint raise VarintError and stay in the buffer for the
    caller to inspect.
    """

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data):
        """Append bytes that arrived from the peer."""
        if isinstance(data, str):
            raise TypeError("feed() takes bytes, not text")
        self._buffer.extend(data)

    def pending(self):
        """Number of buffered bytes that have not been consumed yet."""
        return len(self._buffer)

    def reset(self):
        """Drop every buffered byte."""
        self._buffer = bytearray()

    def read_value(self):
        """Return the next complete value, or None while more bytes are needed."""
        if not self._buffer:
            return None
        try:
            value, size = decode_uvarint(bytes(self._buffer))
        except VarintError as error:
            if error.incomplete:
                return None
            raise
        del self._buffer[:size]
        return value
