import express from "express";
import cookieParser from "cookie-parser";
import authRouter from "./routes/auth.route.js";
import ocrRouter from "./routes/ocr.route.js";
import "dotenv/config";
import cors from "cors";
export const app = express();

app.use(
  cors({
    origin: process.env.FRONTEND_URL || "http://localhost:5173",
    credentials: true,
  }),
);
app.use(express.json());
app.use(cookieParser());
console.log("FRONTEND_URL =", process.env.FRONTEND_URL);
app.get("/", (req, res) => {
  res.send("Hello from backend");
});
app.use("/api/auth", authRouter);
app.use("/api/ocr", ocrRouter);
