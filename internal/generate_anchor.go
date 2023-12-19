package internal

import (
	"errors"
	"math"
	"sort"
	"strconv"
)

// mkAnchors outputs a set of anchors (windows), given a vector
// of widths (ws) and heights (hs) around a center (x_ctr, y_ctr)
func mkAnchors(w, h, xCtr, yCtr float64) []float64 {
	x1 := xCtr - 0.5*(w-1)
	y1 := yCtr - 0.5*(h-1)
	x2 := xCtr + 0.5*(w-1)
	y2 := yCtr + 0.5*(h-1)

	return []float64{x1, y1, x2, y2}
}

// ratioEnums enumerates a set of anchors for each aspect ratio wrt an anchor.
// [x1,y1,x2,y2]
func ratioEnums(anchor []float64, ratios []float64) [][]float64 {
	width, height, xCtr, yCtr := whctrs(anchor)
	size := width * height
	var sizeRatios []float64
	for _, rt := range ratios {
		sizeRatios = append(sizeRatios, size/rt)
	}
	var ws []float64
	for _, sizeRatio := range sizeRatios {
		ws = append(ws, math.Round(math.Sqrt(sizeRatio)))
	}
	var hs []float64
	for i, w := range ws {
		hs = append(hs, math.Round(w*ratios[i]))
	}
	var anchors [][]float64
	for i, w := range ws {
		anchors = append(anchors, mkAnchors(w, hs[i], xCtr, yCtr))
	}
	return anchors
}

// whctrs returns width, height, x center, and y center for an anchor (window).
func whctrs(anchors []float64) (float64, float64, float64, float64) {
	w := anchors[2] - anchors[0] + 1.0
	h := anchors[3] - anchors[1] + 1.0
	xCtr := anchors[0] + 0.5*(w-1)
	yCtr := anchors[1] + 0.5*(h-1)
	return w, h, xCtr, yCtr
}

// scaleEnum enumerates a set of anchors for each scale wrt an anchor.
func scaleEnum(anchor []float64, scales []float64) [][]float64 {
	w, h, xCtr, yCtr := whctrs(anchor)
	var ws []float64
	var hs []float64
	for _, scale := range scales {
		ws = append(ws, w*scale)
		hs = append(hs, h*scale)
	}
	var anchors [][]float64
	for i, _ := range scales {
		anchors = append(anchors, mkAnchors(ws[i], hs[i], xCtr, yCtr))
	}
	return anchors
}

// GenerateAnchors2 generates anchor (reference) windows by enumerating aspect ratios X
// scales wrt a reference (0, 0, 15, 15) window.
func generateAnchors2(baseSize float64, ratios []float64, scales []float64, stride int, denseAnchor bool) ([][]float64, error) {
	baseAnchor := []float64{0, 0, baseSize - 1, baseSize - 1}
	ratioAnchors := ratioEnums(baseAnchor, ratios)
	var anchors [][]float64
	for i := 0; i < len(ratioAnchors); i++ {
		scaleAnchors := scaleEnum(ratioAnchors[i], scales)
		anchors = append(anchors, scaleAnchors...)
	}
	if denseAnchor {
		if stride%2 != 0 {
			return nil, errors.New("stride must be divisible by 2")
		}
		anchors2 := make([][]float64, len(anchors))
		for i := 0; i < len(anchors); i++ {
			anchors2[i] = make([]float64, len(anchors[i]))
			for j := 0; j < len(anchors[i]); j++ {
				anchors2[i][j] = anchors[i][j] + float64(stride)/2
			}
		}
		anchors = append(anchors, anchors2...)
	}
	return anchors, nil
}

// GenerateAnchorsFPN2 generates anchor (reference) windows by enumerating aspect ratios X
// scales wrt a reference (0, 0, 15, 15) window.
func GenerateAnchorsFPN2(denseAnchor bool, cfgs map[string]Anchor) ([][][]float64, error) {
	var anchors [][][]float64
	var RPNFeatStride []int
	for key, _ := range cfgs {
		intKey, err := strconv.Atoi(key)
		if err != nil {
			return nil, err
		}
		RPNFeatStride = append(RPNFeatStride, intKey)
	}

	sort.Slice(RPNFeatStride, func(i, j int) bool {
		return RPNFeatStride[i] > RPNFeatStride[j]
	})

	for key, value := range cfgs {
		baseSize := value.BaseSize
		ratios := value.Ratio
		scales := value.Scale
		stride, err := strconv.Atoi(key)
		if err != nil {
			return nil, err
		}
		anchor, err := generateAnchors2(float64(baseSize), ratios, scales, stride, denseAnchor)
		if err != nil {
			return nil, err
		}
		anchors = append(anchors, anchor)
	}

	return anchors, nil
}
