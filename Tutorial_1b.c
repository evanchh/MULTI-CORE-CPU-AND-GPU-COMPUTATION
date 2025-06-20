#include <stdio.h>
#include <string.h>

void encodeString(char *str) {
    for (int i = 0; str[i] != '\0'; i++) {
        if (strchr("AEIOUaeiou", str[i])) {
            str[i] = 'p';
        }
    }
}

int main() {
    char message[100];

    printf("Enter the string you want to encode:\n");
    fgets(message, sizeof(message), stdin);

    // 移除換行符號
    message[strcspn(message, "\n")] = '\0';

    printf("You typed: %s\n", message);

    encodeString(message);

    printf("Encoded message: %s\n", message);

    return 0;
}

