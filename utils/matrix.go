package utils

import "fmt"

func MakeMatrix[T any](t T, row, col, depth int) {
	detImg := make([][][]uint8, 5)
	for i := 0; i < 5; i++ {
		detImg[i] = make([][]uint8, 4)
		for j := 0; j < 4; j++ {
			detImg[i][j] = make([]uint8, 3)
		}
	}
	fmt.Println("detImg", detImg)

}
