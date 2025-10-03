import { useEffect } from 'react';

export const useRightClickNewTab = () => {
  useEffect(() => {
    const handleContextMenu = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (!target) {
        return;
      }

      const clickable = target.closest('a, button') as (HTMLAnchorElement | HTMLButtonElement | null);
      if (!clickable) {
        return;
      }

      if (clickable instanceof HTMLAnchorElement) {
        if (!clickable.href) {
          return;
        }

        event.preventDefault();
        window.open(clickable.href, '_blank', 'noopener');
        return;
      }

      if (clickable instanceof HTMLButtonElement) {
        if (clickable.disabled) {
          return;
        }

        const rawUrl = clickable.dataset.newTabUrl || window.location.href;

        try {
          const resolvedUrl = new URL(rawUrl, window.location.href).toString();
          event.preventDefault();
          window.open(resolvedUrl, '_blank', 'noopener');
        } catch (error) {
          // If the URL is invalid, allow the default context menu behaviour
        }
      }
    };

    document.addEventListener('contextmenu', handleContextMenu);
    return () => {
      document.removeEventListener('contextmenu', handleContextMenu);
    };
  }, []);
};

export default useRightClickNewTab;
