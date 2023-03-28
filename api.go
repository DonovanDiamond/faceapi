package main

import (
	"encoding/base64"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"

	face "github.com/Kagami/go-face"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/pkg/errors"
)

const dataDir = "data"

var (
	modelsDir                = filepath.Join(dataDir, "models")
	trainingDir              = filepath.Join(dataDir, "training")
	defaultUseCNN            = true
	defaultThreshold float32 = 0.6
)

type FaceAPI struct {
	rec *face.Recognizer
}

func (a *FaceAPI) init() (err error) {
	log.Println("Creating recognizer...")
	a.rec, err = face.NewRecognizer(modelsDir)
	if err != nil {
		return errors.Wrap(err, "failed to create new recognizer")
	}

	log.Println("Creating router...")
	app := fiber.New(fiber.Config{
		BodyLimit: 1024 * 1024 * 1024, // 1GB
	})

	app.Use(logger.New())

	app.Post("/recognize", func(c *fiber.Ctx) error {
		var d struct {
			Data      string   `json:"data"`
			Multiple  bool     `json:"multiple"`
			Threshold *float32 `json:"threshold"`
			UseCNN    *bool    `json:"cnn"`
		}

		if err := c.BodyParser(&d); err != nil {
			return errors.Wrap(err, "failed to parse body")
		}

		raw, err := base64.StdEncoding.DecodeString(d.Data)
		if err != nil {
			return errors.Wrap(err, "failed to decode base64")
		}

		if d.Threshold == nil {
			d.Threshold = &defaultThreshold
		}
		if d.UseCNN == nil {
			d.UseCNN = &defaultUseCNN
		}

		var results []int
		if d.Multiple {
			results, err = a.recongizeMultiple(raw, *d.UseCNN, *d.Threshold)
		} else {
			results = append(results, 0)
			results[0], err = a.recognize(raw, *d.UseCNN, *d.Threshold)
		}

		return c.JSON(fiber.Map{
			"results": results,
			"error":   err,
		})
	})

	app.Post("/add", func(c *fiber.Ctx) error {
		var d struct {
			Data string `json:"data"`
			ID   string `json:"id"`
		}

		if err := c.BodyParser(&d); err != nil {
			return errors.Wrap(err, "failed to parse body")
		}

		raw, err := base64.StdEncoding.DecodeString(d.Data)
		if err != nil {
			return errors.Wrap(err, "failed to decode base64")
		}

		filename := fmt.Sprintf("%s/%d.jpg", d.ID, time.Now().Unix())

		if _, err := os.Stat(filepath.Join(trainingDir, d.ID)); errors.Is(err, os.ErrNotExist) {
			log.Println("Creating Directory...")
			err := os.Mkdir(filepath.Join(trainingDir, d.ID), os.ModePerm)
			if err != nil {
				log.Println(err)
			}
		}
		
		err = os.WriteFile(filepath.Join(trainingDir, filename), raw, fs.ModeAppend)
		if err != nil {
			return errors.Wrap(err, "failed to write output file")
		}

		return c.JSON(fiber.Map{
			"filename": filename,
			"id":       d.ID,
		})
	})

	app.Post("/train", func(c *fiber.Ctx) error {
		err := a.train()

		return c.JSON(fiber.Map{
			"error": err,
		})
	})

	log.Println("Training...")
	err = a.train()
	if err != nil {
		return errors.Wrap(err, "failed to train")
	}

	log.Println("listening...")
	return app.Listen(":1234")
}

func (a *FaceAPI) train() (err error) {
	folders, err := os.ReadDir(trainingDir)
	if err != nil {
		return errors.Wrap(err, "failed to read training dir")
	}
	var samples []face.Descriptor
	var cats []int32

	for _, folder := range folders {
		folderPath := filepath.Join(trainingDir, folder.Name())
		files, err := os.ReadDir(folderPath)
		if err != nil {
			log.Printf("could not read folder %s", folderPath)
			log.Println(err)
			continue
		}
		personID, err := strconv.Atoi(folder.Name())
		if err != nil {
			log.Printf("could not parse folder name %s", folderPath)
			log.Println(err)
			continue
		}

		for i, file := range files {
			if i > 0 {
				continue
			}
			imagePath := filepath.Join(folderPath, file.Name())
			face, err := a.rec.RecognizeSingleFile(imagePath)
			if err != nil {
				log.Printf("could not recognize file %s", imagePath)
				log.Println(err)
				continue
			}
			if face == nil {
				log.Printf("could not find face on %s", imagePath)
				continue
			}
			cats = append(cats, int32(personID))
			samples = append(samples, face.Descriptor)
			log.Printf("\t training %s...", imagePath)
		}
	}
	a.rec.SetSamples(samples, cats)
	return
}

func (a *FaceAPI) recognize(data []byte, useCNN bool, threshold float32) (result int, err error) {
	var face *face.Face
	if useCNN {
		face, err = a.rec.RecognizeSingleCNN(data)
	} else {
		face, err = a.rec.RecognizeSingle(data)
	}
	if err != nil {
		return 0, errors.Wrap(err, "failed to recongize face")
	}

	if face == nil {
		return 0, errors.Wrap(err, "no face on image")
	}
	id := a.rec.ClassifyThreshold(face.Descriptor, threshold)
	return id, nil
}

func (a *FaceAPI) recongizeMultiple(data []byte, useCNN bool, threshold float32) (results []int, err error) {
	var faces []face.Face
	if useCNN {
		faces, err = a.rec.RecognizeCNN(data)
	} else {
		faces, err = a.rec.Recognize(data)
	}
	if err != nil {
		err = errors.Wrap(err, "failed to recongize face")
	}

	for _, face := range faces {
		results = append(results, a.rec.ClassifyThreshold(face.Descriptor, threshold))
	}

	return
}
