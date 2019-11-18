
# Title something

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

## Another Title

__Text__ more text ***More Text***

### Test!

This is a test of *pandoc*.

- list one
- list two

#### Code

```cpp
#include "stdlib.h"
#include <iostream>

void main(){
    std::cout << "Something" << std::endl;
    // Comment
    int thisint = thatint = 7;
    for (int i = 0; i < 10; i++){
        //Something
        std::cout << "Something" << std::endl;
    }
}
```
## Refecences

### This is a heading name

click the link the link [This is a heading name] for more information

## This is another heading name {#header_id}

see Section \ref{header_id} for more information






#### Code with reference


``` {#code_id .cpp}
for (int i = 0; i < 10; i++){
    //Something
    std::cout << "Something" << std::endl;
    do_this();
    do_that();
}
```

See Listing \ref{code_id}.

### Image figure

Some text here

![This is the figure text](./img/image.jpeg){#imagelabel width=50%}

see Figure \ref{imagelabel}

### Image not as a figure

Place non-whitespace character after image

![This is the caption of the image](./img/image.jpeg){ width=50% }\


### Image figure reference

![This is the figure text](img/image.jpeg){#figurelabel width=100 height=50px}

See figure \ref{figurelabel}.

## Something else

<!-- [ref]: img/image.jpeg "optional title" {#id .class key=val key2="val 2"} -->

