import { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Eye, EyeOff } from "lucide-react"; // Icons import
import { InputField } from "./InputField";
import useAuth from "@/hooks/useAuth";

export default function Login() {
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const { handleLogin } = useAuth();

  return (
    <div className="h-full w-full flex items-center justify-center">
      <div
        className="w-full max-w-sm bg-[#f8ffe8] border border-neutral-200 shadow-lg rounded-2xl py-5"
      >
        <div className="text-center pb-2">
          <h1 className="text-2xl font-bold text-neutral-900">Welcome Back</h1>
        </div>

        <div className="p-6 pt-2">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleLogin({ email, password });
            }}
            className="flex flex-col gap-3"
          >
            <InputField
              type="email"
              name="email"
              label="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="name@domain.com"
              autoComplete="off"
              className="w-full p-2.5 text-xs border border-neutral-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-neutral-800"
            />

            <div className="flex items-end relative">
              <InputField
                type={showPassword ? "text" : "password"}
                label="Password"
                name="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                autoComplete="off"
                className="w-full p-2.5 pr-10 text-xs border border-neutral-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-neutral-800"
              />
              <Button
                variant="link"
                size="icon"
                type="button"
                onClick={() => setShowPassword((prev) => !prev)}
                className="absolute right-0"
              >
                {showPassword ? (
                  <EyeOff className="w-4 h-4" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
              </Button>
            </div>

            <Button type="submit" className="bg-[#84994F]">Login</Button>

            <div className="text-center mt-2">
              <p className="text-xs text-zinc-500">
                Don't have an account?{"  "}
                <Link
                  to="/register"
                  className="underline underline-offset-4 cursor-pointer text-black"
                >
                  Register
                </Link>
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
