#include "message_lib.h"

int main() {
    std::cout << echoMessage("Hello, world!") << std::endl;
    
    int sum = addNumbers(2, 3);
    std::cout << "The sum of 2 and 3 is " << sum << std::endl;

    float sum2 = addNumbers((float) 2.5, (float) 3.9);
    std::cout << "The sum of 2.5 and 3.9 is " << sum2 << std::endl;
    return 0;
}