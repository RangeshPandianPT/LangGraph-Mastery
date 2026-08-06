import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
});

const Mermaid = ({ chart }) => {
  const ref = useRef(null);

  useEffect(() => {
    if (chart && ref.current) {
      mermaid.render('mermaid-svg', chart).then((result) => {
        ref.current.innerHTML = result.svg;
      }).catch((e) => {
        console.error("Mermaid parsing error", e);
      });
    }
  }, [chart]);

  return <div ref={ref} className="mermaid-chart"></div>;
};

export default Mermaid;
