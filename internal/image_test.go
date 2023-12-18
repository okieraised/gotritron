package internal

import (
	"fmt"
	"io"
	"os"
	"testing"
)

func TestConvertImageToOpenCV(t *testing.T) {
	f, err := os.Open("./test_data/harrison.jpeg")
	if err != nil {
		fmt.Println(err)
		return
	}

	content, err := io.ReadAll(f)
	if err != nil {
		fmt.Println(err)
		return
	}

	res, err := ConvertImageToOpenCV(content)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res)
}
