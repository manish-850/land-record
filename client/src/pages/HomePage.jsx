import { useNavigate } from "react-router-dom";
// import card1Img from "../assets/heroimg1.png";
// import card2Img from "@/assets/heroimg2.webp";
// import card2Img from "../assets/heroimg2.webp";
// import card3Img from "../assets/heroimg3.png";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/home/Navbar";
import ImgCard from "@/components/home/ImgCard";

export default function HomePage() {
  const navigate = useNavigate();
  return (
    <main className="h-screen w-full bg-white text-neutral-900 flex flex-col items-center px-50 py-5">
      <Navbar />
      <div className="w-full flex flex-row items-center justify-between flex-1">
        <div className="flex-1 flex flex-col items-start justify-between z-10 gap-4">
          <h2 className="text-5xl font-bold leading-[1.08] text-[#84994F]">
            Land Record <br />
            Digitizer
          </h2>

          <p className="text-sm leading-[1.6] text-neutral-600 w-[70%]">
            Simplifying land record administration through automated document
            processing, secure data extraction, and hassle-free status tracking.
          </p>

          <div className="flex gap-4">
            <Button
              className="bg-[#84994F]"
              type="button"
              onClick={() => navigate("/login")}
            >
              Login
            </Button>
            <Button
              className="bg-[#d2e99c]"
              type="button"
              variant="secondary"
              onClick={() => navigate("/register")}
            >
              Create Account
            </Button>
          </div>
        </div>

        <div className="relative flex items-center justify-center">
          <ImgCard
            src={
              "https://i.pinimg.com/1200x/63/ad/4b/63ad4bba93fff9f3ee7224a65b493cf4.jpg"
            }
            alt="Cadastral Map"
            className="w-50 h-90 z-1 shadow-[0_8px_16px_-4px_rgba(0,0,0,0.1)]"
          />
          <ImgCard
            src={
              "https://i.pinimg.com/1200x/eb/a2/e5/eba2e5e96e59e6aa6579f8a467aab52d.jpg"
            }
            alt="Document Digitization"
            className="w-55 h-100 z-2 shadow-[0_22px_35px_-8px_rgba(0,0,0,0.18)] -ml-10 grayscale"
          />
          <ImgCard
            src={
              "https://i.pinimg.com/1200x/c0/a5/4e/c0a54e27a98e9364782e2aa364522782.jpg"
            }
            alt="Data Dashboard"
            className="w-50 h-90 z-1 shadow-[0_8px_16px_-4px_rgba(0,0,0,0.1)] -ml-10"
          />
        </div>
      </div>
    </main>
  );
}
