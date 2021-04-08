# Scrolling text on pico scroll (redux)

## TL;DR

Following on from [a Python implementation](./2021-04-06-Scoller), wouldn't it be nice to have text scrolling built into the picoscroll micropython API?

[Let's ask how they feel about that](https://github.com/pimoroni/pimoroni-pico/pull/121)


## verbose=true

Given that this is called a pico scroll you could still think that scrolling text is literally it's raison d'être right? So offer them a couple of methods to (i) simply blit an image to the lights and (ii) scroll a bitmap across the display where you've carefully arranged a `bytearray` with which pixels should on / off, and want to view a 17 pixel wide slice of this... 

This was something of a learning process as it involved adding C / C++ code to the micropython firmware, extending the Pimoroni API extensions further.

### Learning

A lot of micropython is held together with preprocessor macros, so you have to add a C definition like

```C
STATIC MP_DEFINE_CONST_FUN_OBJ_3(picoscroll_show_bitmap_1d_obj, picoscroll_show_bitmap_1d);
```

which explains to the compiler that this is a function with 3 arguments. You then need to add the _name_ of this function to the list of strings that the micropython interpreter knows about:

```C
{ MP_ROM_QSTR(MP_QSTR_show_bitmap_1d), MP_ROM_PTR(&picoscroll_show_bitmap_1d_obj) },
```

as micropython has some aggressive methods to save memory (rather reasonably). The actual code added is pretty simple:

```C
mp_obj_t picoscroll_show_bitmap_1d(mp_obj_t bitmap_obj, mp_obj_t brightness_obj, mp_obj_t offset_obj) {
    if(scroll != nullptr) {
        mp_buffer_info_t bufinfo;
	mp_get_buffer_raise(bitmap_obj, &bufinfo, MP_BUFFER_RW);
        int offset = mp_obj_get_int(offset_obj);
        int brightness = mp_obj_get_int(brightness_obj);
	int length = bufinfo.len;
	int width = PicoScroll::WIDTH;
	int height = PicoScroll::HEIGHT;

	// this obviously shouldn't happen as the scroll is 17x7 pixels
	// would fall off end of byte if this the case
	if (height > 8) {
	    mp_raise_msg(&mp_type_RuntimeError, INCORRECT_SIZE_MSG);
	}

	unsigned char * values = (unsigned char *) bufinfo.buf;

	// clear the scroll, so only need to write visible bytes
	scroll->clear();

	if ((offset < -width) || (offset > length)) {
	    return mp_const_none;
	}

	for (int x = 0; x < width; x++) {
	    int k = offset + x;
	    if ((k >= 0) && (k <= length)) {
	        unsigned char col = values[k];
		for (int y = 0; y < height; y++) {
		    int val = brightness * ((col >> y) & 1);
		    scroll->set_pixel(x, height - 1 - y, val);
		}
	    }
	}
    } else {
        mp_raise_msg(&mp_type_RuntimeError, NOT_INITIALISED_MSG);
    }

    return mp_const_none;
}
```

Unlike some methods of extending languages, most of the code here is actually doing the things I want to do, which shows some good and quite compact design. Unpacking the input objects, trying to get the `bytearray` buffer and verifying that the input is sane is the top half, then selecting the view of the buffer to display - if this is empty, do nothing, else show the bits that you can see (literally) using bit shift operators.

This is a plug in replacement for the simple Python API as
demonstrated by
[this gist](https://gist.github.com/graeme-winter/ff08123ceae76399791413f2564eecaa)
with the output like [this](https://youtu.be/XIvKc523NwM). Additional:
you can get the
[firmware for this here](https://github.com/graeme-winter/rpi-pico/raw/main/firmware/picoscroll-micropython-plus-2021-04-08.uf2). 
