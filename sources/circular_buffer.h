#pragma once

#include <memory>
#include <mutex>

template <class T>
class CircularBuffer {
public:
	explicit CircularBuffer(size_t size) :
		_buf(std::unique_ptr<T[]>(new T[size])),
		_head(0),
		_tail(0),
		_max_size(size),
		_full(false) {}

	void put(T item) {
		std::lock_guard<std::mutex> lock(_mutex);
		_buf[_head] = item;

		if (_full) {
			_tail = (_tail + 1) % _max_size;
		}

		_head = (_head + 1) % _max_size;
		_full = (_head == _tail) ? true : false;
	}

	T get() {
		std::lock_guard<std::mutex> lock(_mutex);

		if (empty()) {
			return T();
		}

		//Read data and advance the tail (we now have a free space)
		auto val = _buf[_tail];
		_full = false;
		_tail = (_tail + 1) % _max_size;

		return val;
	}

	void reset() {
		std::lock_guard<std::mutex> lock(_mutex);
		_head = _tail;
		_full = false;
	}

	bool empty() const {
		//if head and tail are equal, we are empty
		return (!_full && (_head == _tail));
	}

	bool full() const {
		//If tail is ahead the head by 1, we are full
		return _full;
	}

	size_t capacity() const {
		return _max_size;
	}

	size_t size() const {
		size_t size = _max_size;

		if (!_full) {
			if (_head >= _tail) {
				size = _head - _tail;
			} else {
				size = _max_size + _head - _tail;
			}
		}

		return size;
	}

private:
	std::mutex _mutex;
	std::unique_ptr<T[]> _buf;
	size_t _head;
	size_t _tail;
	const size_t _max_size;
	bool _full;
};
