import { Router } from "express";
import { extractDataController } from "../controllers/ocr.controller.js";
import multer from "multer";

const upload = multer({
    storage: multer.memoryStorage()
});

const ocrRouter = Router();

ocrRouter.post("/extract", upload.single("file"), extractDataController);

export default ocrRouter;
