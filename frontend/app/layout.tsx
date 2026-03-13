import type { Metadata } from "next";
import Nav from "@/components/nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "YOHAS 3.0",
  description: "Your Own Hypothesis-driven Agentic Scientist",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased">
        <Nav />
        <main>{children}</main>
      </body>
    </html>
  );
}
