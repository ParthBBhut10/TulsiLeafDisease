import Breadcrumb from "@/components/Common/Breadcrumb";
import Contact from "@/components/Contact";

import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Contact Page | Tulsi Leaf Disease",
  description: "Disease Detection",
  // other metadata
};

const ContactPage = () => {
  return (
    <>
      <Breadcrumb
        pageName="Contact"
        description="Detect Tulsi leaf disease using image recognition software for quick and accurate diagnosis and treatment."
      />

      <Contact />
    </>
  );
};

export default ContactPage;
