import { SearchBar } from '../../shared/ui/SearchBar/SearchBar';
import cls from './AdminUsersToolbar.module.scss';

interface AdminUsersToolbarProps {
  searchValue: string;
  onSearchChange: (value: string) => void;
  onSearchSubmit: () => void;
}

export const AdminUsersToolbar = ({
  searchValue,
  onSearchChange,
  onSearchSubmit,
}: AdminUsersToolbarProps) => {
  return (
    <div className={cls.toolbar}>
      <SearchBar
        value={searchValue}
        onChange={onSearchChange}
        onSearch={onSearchSubmit}
        placeholder="Поиск пользователя..."
      />
    </div>
  );
};