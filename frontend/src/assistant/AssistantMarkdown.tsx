import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function AssistantMarkdown({ children }: { children: string }) {
  return <div className="assistant-response">
    <span className="sr-only">Assistant: </span>
    <Markdown
      remarkPlugins={[remarkGfm]}
      skipHtml
      components={{
        a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
        table: ({ children }) => <div className="assistant-table"><table>{children}</table></div>,
      }}
    >{children}</Markdown>
  </div>;
}
