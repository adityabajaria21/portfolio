export default function ProjectCard({ project, onViewProject }) {
  return (
    <div className="bg-white rounded-lg p-4 shadow">
      <h3>{project.title}</h3>
      <button onClick={() => onViewProject(project.id)}>View</button>
    </div>
  );
}
