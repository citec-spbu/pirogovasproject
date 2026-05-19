import type { KeyboardEvent } from 'react';
import searchIcon from '../../assets/icons/searchIcon.svg';
import cls from './SearchBar.module.scss';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSearch?: () => void;
  placeholder?: string;
  className?: string;
}

export const SearchBar = ({
  value,
  onChange,
  onSearch,
  placeholder = 'Поиск...',
  className = '',
}: SearchBarProps) => {
  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      onSearch?.();
    }
  };

  return (
    <div className={[cls.searchBar, className].filter(Boolean).join(' ')}>
      <div className={cls.inputWrapper}>
        <img src={searchIcon} alt="" className={cls.searchIcon} />

        <input
          className={cls.input}
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
        />
      </div>

      <button type="button" className={cls.searchButton} onClick={onSearch}>
        Поиск
      </button>
    </div>
  );
};