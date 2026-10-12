use std::{marker::PhantomData, ops::Deref, ptr::NonNull};

/// A compressed, read-only [`Cow<'a, str>`] that holds at most `u31::MAX` bytes.
pub struct GraphemeStr<'a> {
    ptr: NonNull<u8>,
    len: u32,
    _marker: PhantomData<&'a str>,
}

impl GraphemeStr<'_> {
    const MASK_OWNED: u32 = 1 << 31;

    fn len(&self) -> usize {
        (self.len & !Self::MASK_OWNED) as usize
    }
}

impl Deref for GraphemeStr<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        unsafe {
            let bytes = core::slice::from_raw_parts(self.ptr.as_ptr(), self.len());
            str::from_utf8_unchecked(bytes)
        }
    }
}

impl Drop for GraphemeStr<'_> {
    fn drop(&mut self) {
        // Ensure owned memory gets freed.
        if self.len & Self::MASK_OWNED != 0 {
            unsafe {
                let obj = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    self.ptr.as_ptr(),
                    self.len(),
                ));
                drop(obj)
            }
        }
    }
}

impl<'a> From<&'a str> for GraphemeStr<'a> {
    fn from(value: &'a str) -> Self {
        GraphemeStr {
            ptr: unsafe { NonNull::new_unchecked(value.as_bytes().as_ptr() as *mut u8) },
            // i32 try_from ensures the len cannot exceed u31::MAX
            len: i32::try_from(value.len()).unwrap() as u32,
            _marker: PhantomData,
        }
    }
}

impl From<String> for GraphemeStr<'_> {
    fn from(value: String) -> Self {
        let len = value.len();
        let ptr = Box::into_raw(value.into_bytes().into_boxed_slice()) as *mut u8;
        GraphemeStr {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len: (i32::try_from(len).unwrap() as u32) | Self::MASK_OWNED,
            _marker: PhantomData,
        }
    }
}
