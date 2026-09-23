import type { Metadata } from "next";
import { Toaster } from "../components/ui/toaster";
import "./globals.css";

export const metadata: Metadata = {
  title: "VAPT-AI",
  description: "Vulnerability Assessment & Penetration Testing AI — multi-agent pentest console",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        {children}
        <Toaster />
      </body>
    </html>
  );
}
